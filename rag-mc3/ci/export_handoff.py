#!/usr/bin/env python3
"""Export only a small audited AMD JSON handoff; no weights/vendor ZIP."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform


def vendor_identity(rows):
    selected = [{key: row[key] for key in ('path', 'bytes', 'mode', 'sha256')}
                for row in rows if row['kind'] == 'file' and row['path'].startswith('app/vendor/')]
    selected.sort(key=lambda row: row['path'])
    encoded = json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()
    return {'files': len(selected), 'bytes': sum(row['bytes'] for row in selected),
            'sha256': hashlib.sha256(encoded).hexdigest(),
            'method': 'sorted path,bytes,mode,sha256 tuples; independent of source paths and mtimes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Export this proof on AMD after its native import smoke and GPU quality gate')
    spec = importlib.util.spec_from_file_location('ci_layer_plan', pathlib.Path(__file__).with_name('stream_hf_layer.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.plan(args.stage)
    receipt = json.loads((args.stage / 'APP-LAYER-RECEIPT.json').read_text())
    quality = json.loads((args.stage / 'GPU-QUALITY-GATE.json').read_text())
    image_config = json.loads((args.stage / 'IMAGE-CONFIG.json').read_text())
    model_files = [{key: row[key] for key in ('path', 'kind', 'bytes', 'mode', 'sha256')}
                   for row in manifest if row['kind'] == 'file' and row['path'].startswith('models/qwen/')]
    # The original complete operational proof stays private in the AMD stage.
    # Public CI needs only content identities and exact package/model contracts:
    # no host paths, commands, emails, device telemetry or raw query results.
    smoke = receipt['native_vendor_import_smoke']
    public_smoke = {key: smoke[key] for key in ('torch_version', 'torch_hip', 'model_loaded', 'gpu_inference') if key in smoke}
    public_smoke['source'] = 'preserved fresh AMD base-Python imports with exact audited vendor; no model/inference'
    public_smoke['final_container_imports_verified'] = False
    public_receipt = {key: receipt[key] for key in ('status', 'weights_included', 'weights_bytes', 'model_revision', 'dependency_closure_verified')}
    public_receipt['native_vendor_import_smoke'] = public_smoke
    public_receipt['dependency_closure'] = {
        'requirements_sha256': receipt['dependency_closure']['requirements_sha256'],
        'wheels': [{key: row[key] for key in ('name', 'version')} for row in receipt['dependency_closure']['wheels']]}
    public_receipt['wheels'] = [{key: row[key] for key in ('name', 'version', 'filename', 'sha256', 'source_url', 'bytes', 'tags', 'requires_python') if key in row}
                              for row in receipt['wheels']]
    public_receipt['runtime_imports_verified_in_final_image'] = False
    public_quality = {key: quality[key] for key in ('quality_verified', 'queries_total', 'queries_passed', 'runtime_sha256')}
    handoff = {'status': 'verified_amd_runtime_handoff', 'handoff_version': 1,
               'model_revision': module.REVISION, 'model_manifest': model_files,
               'vendor_identity': vendor_identity(manifest), 'gpu_quality_gate': public_quality,
               'app_layer_receipt': public_receipt, 'image_config': image_config,
               'model_bytes_exported': 0, 'vendor_bytes_exported': 0,
               'final_container_gpu_execution_verified': False}
    data = (json.dumps(handoff, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(data) > 256 * 1024:
        parser.error('Handoff exceeds the 256 KiB small JSON budget; no file was created')
    if args.output.exists() or args.output.is_symlink() or any(p.is_symlink() for p in args.output.parents):
        parser.error('Use a new handoff filename; existing evidence is preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        output.write(data)
    print(json.dumps({'status': 'small_json_handoff_created', 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(), 'model_bytes_exported': 0,
                      'vendor_bytes_exported': 0, 'registry_contacted': False}))


if __name__ == '__main__':
    main()
