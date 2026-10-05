# Figueira RAG MC3 — review snapshot

This source answers document questions with a persistent AMD GPU worker, a locally indexed corpus, and JSON answers with citations. It is being prepared for the AMD/LabLab MC3 challenge.

## Confirmed native evidence and remaining image work

The frozen R05 source passed **14 of 14** official and independent queries in the native AMD GPU pilot. Combined model/index startup took **37.378 seconds**, and the complete query phase took **39.812 seconds**. Recorded peak total GPU VRAM used was **18,212,052,992 bytes**. The pilot confirmed that its worker stopped after testing. Local CPU tests reported **23 of 23** passing. The R05 `rag_core.py` SHA256 is `d299f958a9dbfd562f4820bfa1b792a7a2e07514129e53192e444c39f81e57a8`.

Native vendor imports passed on the AMD base runtime. That proves the staged dependencies imported in that runtime; it does not prove execution inside the final OCI image. **Final image assembly, anonymous pull checks, and final-container GPU execution remain pending.** The actual sanitized AMD handoff is being prepared for this source; no placeholder proof is included in this snapshot. The CI job rejects a missing handoff, a failed quality gate, or runtime bytes that differ from the completed GPU pilot.

## Runtime contract

Run `python3 /app/app.py --index /app/corpus` once. Each question then uses a new CLI process:

```sh
python3 /app/app.py --corpus /app/corpus --query-id ID --query "Question text"
```

The worker remains resident through a local Unix socket. Answers are written to `/app/output/ID_output.json` with `answer` and `citations`. Unsupported questions abstain with an empty answer and citation list. The application uses pinned local Qwen2.5-VL-7B-Instruct weights; it must not download at evaluation time. Corpus files and expected answers are evaluation inputs, not part of the published source or image.

## Planned offline OCI package

The mandatory base is `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, pinned for reproducibility to `sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`. Its original 11 layers and diff IDs must remain unchanged. The planned package appends one application/model layer; it is OCI assembly, not a Dockerfile build or a squashed base.

The model is `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. All five safetensor shards and model configuration must be embedded. `RAG_ALLOW_MODEL_DOWNLOAD=0`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` are required. No hosted model endpoint is used.

The manual workflow is intended for a free standard Linux runner in a public repository. It reconstructs the exact native-tested vendor from official, hash-verified PyPI wheels and streams the pinned model from its official Hugging Face revision without caching weights on disk or transferring them to the Mac. It mounts the base layers within Docker Hub; the assembler rejects a base-layer download or a total decompressed size above 60 GiB before publishing. The final image reference is supplied privately through `RAG_IMAGE_REF`; it must not be posted in this repository or a public announcement.

Before a run, add only the **actual completed** sanitized proof as `evidence/AMD-HANDOFF.json`. The workflow also requires its exact SHA256 and the exact approved source commit. The handoff must bind all seven runtime files to a genuine 14/14 GPU result, the complete wheel closure, the exact audited vendor identity, and previously verified model hashes. It contains no weights, credentials, raw queries, expected answers, private host telemetry, or final image reference.

Docker Hub authentication uses the ordinary one-time device login of the authorized owner within that runner. No password, PAT, Mac Docker configuration, or Keychain data is copied. A job must be explicitly dispatched and authorized; this snapshot starts no jobs. The runner login is temporary and logged out on completion. No paid runner, GPU job, Docker Build Cloud, or paid service is selected.

## Verification limits

CPU fixtures verify helper gates, paths, hashes, safe extraction, replay identity, and stream integrity. They do not replace AMD GPU quality testing. Native GPU tests of the source do not prove offline startup or inference inside the final published image. The receipt must retain those distinctions, and submission should use only the resulting confirmed image reference through the authenticated challenge form.
