// Remote-only OCI assembly: stream the app and prepared weights, mount the base.
// Auth uses the ordinary Docker keychain; no secret is printed or exported.
// Do not enable tarball.WithCompressedCaching: weights must not enter RAM in full.
package main

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "os"
    "os/exec"
    "path"
    "path/filepath"
    "runtime"
    "strings"
    "sync/atomic"
    "time"

    "github.com/google/go-containerregistry/pkg/authn"
    "github.com/google/go-containerregistry/pkg/name"
    v1 "github.com/google/go-containerregistry/pkg/v1"
    "github.com/google/go-containerregistry/pkg/v1/mutate"
    "github.com/google/go-containerregistry/pkg/v1/remote"
    "github.com/google/go-containerregistry/pkg/v1/tarball"
)

const expectedBase = "sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c"
// Exact streamed size receipt for this immutable base digest, preserved in OCR R03.
const measuredBaseUncompressed int64 = 31190284288
const modelRevision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

// Fail closed before hashing/upload if a weights-free or incomplete stage is
// supplied. The Python producer additionally verifies each pinned model shard.
func verifyPreparedStage(stage string) error {
    data,e:=os.ReadFile(filepath.Join(stage,"APP-LAYER-RECEIPT.json"));if e!=nil{return e}
    var receipt struct {
        Status string `json:"status"`
        WeightsIncluded bool `json:"weights_included"`
        WeightsBytes int64 `json:"weights_bytes"`
        ModelRevision string `json:"model_revision"`
        Closure bool `json:"dependency_closure_verified"`
        Smoke map[string]any `json:"native_vendor_import_smoke"`
    }
    if e=json.Unmarshal(data,&receipt);e!=nil{return e}
    if receipt.Status!="stream_plan_prepared"||!receipt.WeightsIncluded||receipt.WeightsBytes<=0||receipt.ModelRevision!=modelRevision||!receipt.Closure {return errors.New("complete pinned offline model and dependency closure receipt required")}
    if receipt.Smoke==nil||receipt.Smoke["model_loaded"]!=false||receipt.Smoke["gpu_inference"]!=false {return errors.New("native vendor import smoke receipt required; no GPU/container inference implied")}
    data,e=os.ReadFile(filepath.Join(stage,"APP-LAYER-FILES.json"));if e!=nil{return e}
    var files []struct {Path string `json:"path"`;Kind string `json:"kind"`;Bytes int64 `json:"bytes"`;SHA string `json:"sha256"`}
    if e=json.Unmarshal(data,&files);e!=nil{return e}
    total:=int64(0);shards:=0;seen:=map[string]bool{};required:=map[string]bool{"models/qwen/config.json":false,"models/qwen/tokenizer_config.json":false,"models/qwen/model.safetensors.index.json":false}
    for _,f:=range files {
        if path.IsAbs(f.Path)||path.Clean(f.Path)!=f.Path||strings.HasPrefix(f.Path,"../")||seen[f.Path] {return errors.New("unsafe/duplicate prepared manifest path")};seen[f.Path]=true
        if f.Kind!="file"&&f.Kind!="directory" {return errors.New("only regular files/directories may enter the image")}
        if !strings.HasPrefix(f.Path,"models/qwen/") {continue}
        if f.Kind!="file" {continue}
        if f.Bytes<=0||len(f.SHA)!=64 {return errors.New("model file needs positive bytes and hash")};total+=f.Bytes
        if _,ok:=required[f.Path];ok{required[f.Path]=true}
        if strings.HasSuffix(f.Path,".safetensors"){shards++}
    }
    if total!=receipt.WeightsBytes||shards!=5 {return errors.New("pinned Qwen model manifest must contain all five weight shards and match receipt bytes")}
    for _,present:=range required {if !present{return errors.New("pinned model config/index/tokenizer manifest is incomplete")}}
    return nil
}

type metadataOnlyLayer struct { v1.Layer }
func (l metadataOnlyLayer) Compressed() (io.ReadCloser, error) { return nil, errors.New("base mount refused: downloading the large base is disabled") }
func (l metadataOnlyLayer) Uncompressed() (io.ReadCloser, error) { return nil, errors.New("base decompression is disabled") }
type guardedBase struct { v1.Image; ref name.Reference }
func (b guardedBase) guarded(l v1.Layer) v1.Layer {
    if m, ok := l.(*remote.MountableLayer); ok { return &remote.MountableLayer{Layer:metadataOnlyLayer{m.Layer}, Reference:m.Reference} }
    return &remote.MountableLayer{Layer:metadataOnlyLayer{l}, Reference:b.ref}
}
func (b guardedBase) Layers() ([]v1.Layer,error) {
    layers,err:=b.Image.Layers(); if err!=nil{return nil,err}; out:=make([]v1.Layer,len(layers))
    for i,l:=range layers {out[i]=b.guarded(l)}; return out,nil
}
func (b guardedBase) LayerByDigest(h v1.Hash) (v1.Layer,error) {l,e:=b.Image.LayerByDigest(h);if e!=nil{return nil,e};return b.guarded(l),nil}
func (b guardedBase) LayerByDiffID(h v1.Hash) (v1.Layer,error) {l,e:=b.Image.LayerByDiffID(h);if e!=nil{return nil,e};return b.guarded(l),nil}

type commandReader struct { io.ReadCloser; cmd *exec.Cmd; bytes int64; measured *atomic.Int64; completed bool }
func (r *commandReader) Read(p []byte) (int,error) {
    n,err:=r.ReadCloser.Read(p);r.bytes+=int64(n)
    if err==io.EOF && !r.completed {
        r.completed=true
        if e:=r.cmd.Wait();e!=nil{return n,fmt.Errorf("TAR stream generator failed: %w",e)}
        r.measured.Store(r.bytes)
    }
    return n,err
}
func (r *commandReader) Close() error {
    e:=r.ReadCloser.Close()
    if !r.completed { _=r.cmd.Process.Kill();_=r.cmd.Wait();r.completed=true }
    return e
}
func writeJSON(path string, value any) error {d,e:=json.MarshalIndent(value,"","  ");if e!=nil{return e};return os.WriteFile(path,append(d,'\n'),0644)}
func fail(e error) {fmt.Fprintln(os.Stderr,e);os.Exit(1)}
func mergeEnv(base,addition []string) []string {
    replaced:=map[string]bool{};for _,s:=range addition {replaced[strings.SplitN(s,"=",2)[0]]=true}
    out:=[]string{};for _,s:=range base {if !replaced[strings.SplitN(s,"=",2)[0]]{out=append(out,s)}}
    return append(out,addition...)
}

func main() {
    // Source compilation can occur elsewhere, but weights/assembly run on Linux.
    if runtime.GOOS!="linux" {fail(errors.New("assemble and publish only on remote Linux; no model bytes on the Mac"))}
    if len(os.Args)<7 {fail(errors.New("usage: assembler BASE PREPARE_SCRIPT STAGE DEST OUTPUTDIR CREATED [--push]"))}
    baseName,script,stage,destination,outdir,createdString:=os.Args[1],os.Args[2],os.Args[3],os.Args[4],os.Args[5],os.Args[6]
    if e:=verifyPreparedStage(stage);e!=nil{fail(e)}
    created,e:=time.Parse(time.RFC3339,createdString);if e!=nil{fail(e)}
    baseRef,e:=name.ParseReference(baseName);if e!=nil{fail(e)}
    destRef,e:=name.ParseReference(destination);if e!=nil{fail(e)}
    ctx,cancel:=context.WithTimeout(context.Background(),75*time.Minute);defer cancel()
    opts:=[]remote.Option{remote.WithContext(ctx),remote.WithAuthFromKeychain(authn.DefaultKeychain),remote.WithPlatform(v1.Platform{OS:"linux",Architecture:"amd64"}),remote.WithJobs(1)}
    base,e:=remote.Image(baseRef,opts...);if e!=nil{fail(e)}
    baseDigest,e:=base.Digest();if e!=nil{fail(e)}
    if baseDigest.String()!=expectedBase {fail(errors.New("base digest changed; revalidate the base contract and measured size before assembly"))}
    baseConfig,e:=base.ConfigFile();if e!=nil{fail(e)}
    baseManifest,e:=base.Manifest();if e!=nil{fail(e)}
    if baseConfig.OS!="linux"||baseConfig.Architecture!="amd64"||len(baseManifest.Layers)!=11 {fail(errors.New("unexpected required base platform/layer count"))}
    configData,e:=os.ReadFile(filepath.Join(stage,"IMAGE-CONFIG.json"));if e!=nil{fail(e)}
    var wanted v1.Config;if e=json.Unmarshal(configData,&wanted);e!=nil{fail(e)}
    if strings.Join(wanted.Cmd," ")!="python3 /app/worker.py"||wanted.WorkingDir!="/app" {fail(errors.New("unexpected worker command"))}
    joined:=strings.Join(wanted.Env,"\n")
    for _,s:=range []string{"RAG_ALLOW_MODEL_DOWNLOAD=0","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1","PYTHONPATH=/app/vendor"} {if !strings.Contains("\n"+joined+"\n","\n"+s+"\n"){fail(fmt.Errorf("offline image config required: %s",s))}}
    var measured atomic.Int64
    opener:=func()(io.ReadCloser,error){
        cmd:=exec.CommandContext(ctx,"python3",script,"--stage",stage,"--stream-tar")
        cmd.Stderr=os.Stderr
        stdout,e:=cmd.StdoutPipe();if e!=nil{return nil,e};if e=cmd.Start();e!=nil{return nil,e}
        return &commandReader{ReadCloser:stdout,cmd:cmd,measured:&measured},nil
    }
    // Default compression is a streaming, fast gzip. Never cache compressed bytes.
    app,e:=tarball.LayerFromOpener(opener);if e!=nil{fail(e)}
    appDiffID,e:=app.DiffID();if e!=nil{fail(e)}
    appDigest,e:=app.Digest();if e!=nil{fail(e)}
    appCompressed,e:=app.Size();if e!=nil{fail(e)}
    appUncompressed:=measured.Load();if appUncompressed<=0{fail(errors.New("complete TAR stream byte count not recorded"))}
    if measuredBaseUncompressed+appUncompressed > 60*1024*1024*1024 {fail(errors.New("measured base plus new layer exceeds 60 GiB; refusing publication"))}
    image,e:=mutate.Append(guardedBase{base,baseRef},mutate.Addendum{Layer:app,History:v1.History{Created:v1.Time{Time:created},CreatedBy:"Figueira RAG: audited app and pinned prepared weights; OCI streamed layer"}});if e!=nil{fail(e)}
    config,e:=image.ConfigFile();if e!=nil{fail(e)};config=config.DeepCopy()
    config.Created=v1.Time{Time:created};config.Config.Cmd=wanted.Cmd;config.Config.Entrypoint=nil;config.Config.WorkingDir=wanted.WorkingDir
    config.Config.Env=mergeEnv(config.Config.Env,wanted.Env);config.Config.Healthcheck=wanted.Healthcheck
    if config.Config.Labels==nil{config.Config.Labels=map[string]string{}}
    config.Config.Labels["org.opencontainers.image.title"]="Figueira RAG"
    config.Config.Labels["org.opencontainers.image.base.digest"]=baseDigest.String()
    config.Config.Labels["figueira.assembly.method"]="streamed OCI append with embedded weights; no Dockerfile build"
    image,e=mutate.ConfigFile(image,config);if e!=nil{fail(e)}
    manifest,e:=image.Manifest();if e!=nil{fail(e)}
    if len(manifest.Layers)!=12||len(config.RootFS.DiffIDs)!=12{fail(errors.New("required original layers were not preserved"))}
    for i,l:=range baseManifest.Layers {if manifest.Layers[i].Digest!=l.Digest||manifest.Layers[i].Size!=l.Size||config.RootFS.DiffIDs[i]!=baseConfig.RootFS.DiffIDs[i]{fail(errors.New("base layer/diff ID prefix differs"))}}
    if config.RootFS.DiffIDs[11]!=appDiffID{fail(errors.New("application diff ID differs"))}
    digest,e:=image.Digest();if e!=nil{fail(e)}
    if e=os.MkdirAll(outdir,0755);e!=nil{fail(e)}
    if e=writeJSON(filepath.Join(outdir,"OCI-CONFIG-PREVIEW.json"),config);e!=nil{fail(e)}
    if e=writeJSON(filepath.Join(outdir,"OCI-MANIFEST-PREVIEW.json"),manifest);e!=nil{fail(e)}
    receipt:=map[string]any{"method":"streamed OCI append; not Dockerfile build","base":baseName,"base_digest":baseDigest.String(),"base_layers_preserved":11,"total_layers":12,"base_transfer_disabled":true,"destination":destination,"manifest_digest":digest.String(),"app_diff_id":appDiffID.String(),"app_digest":appDigest.String(),"app_uncompressed_bytes":appUncompressed,"app_compressed_bytes":appCompressed,"base_uncompressed_bytes":measuredBaseUncompressed,"total_uncompressed_bytes":measuredBaseUncompressed+appUncompressed,"uncompressed_size_limit_verified":true,"model_weights_embedded":true,"runtime_download_disabled":true,"tar_bytes_saved_to_disk":0,"full_compressed_layer_cached_in_ram":false,"push_verified":false,"final_container_gpu_execution_verified":false}
    if e=writeJSON(filepath.Join(outdir,"OCI-ASSEMBLY-RECEIPT.json"),receipt);e!=nil{fail(e)}
    if len(os.Args)>7&&os.Args[7]=="--push" {
        fmt.Println("Publishing the streamed remote application/model layer; base transfers remain blocked")
        if e=remote.Write(destRef,image,opts...);e!=nil{fail(e)}
        receipt["push_verified"]=true
        // Anonymous post-push inspection checks manifest, config, original prefix.
        anonymous,e:=remote.Image(destRef,remote.WithContext(ctx),remote.WithAuth(authn.Anonymous));if e!=nil{fail(e)}
        actual,e:=anonymous.Digest();if e!=nil{fail(e)};if actual!=digest{fail(errors.New("anonymous published digest differs"))}
        ac,e:=anonymous.ConfigFile();if e!=nil{fail(e)}
        if len(ac.RootFS.DiffIDs)!=12||ac.RootFS.DiffIDs[11]!=appDiffID{fail(errors.New("anonymous config diff IDs differ"))}
        for i,d:=range baseConfig.RootFS.DiffIDs {if ac.RootFS.DiffIDs[i]!=d{fail(errors.New("anonymous base diff ID prefix differs"))}}
        receipt["anonymous_manifest_config_verified"]=true
        if e=writeJSON(filepath.Join(outdir,"OCI-CONFIG-ANONYMOUS.json"),ac);e!=nil{fail(e)}
    }
    if e=writeJSON(filepath.Join(outdir,"OCI-ASSEMBLY-RECEIPT.json"),receipt);e!=nil{fail(e)}
    d,_:=json.Marshal(receipt);fmt.Println(string(d))
}
