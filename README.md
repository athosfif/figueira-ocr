# Figueira OCR

A persistent Qwen2.5-VL-7B-Instruct worker for text extraction from vehicle plates and road signs on AMD ROCm. It accepts PNG, JPEG and TIFF and writes an atomic JSON file containing a `text` string. The AMD path requires an AMD GPU; it has no CPU fallback.

The foreground worker acquires a lifetime lock, loads the model, then reports readiness. Separate CLI processes wait for its socket and reuse the model. Inference is serialized. The container preserves and checks the ROCm PyTorch and torchvision versions supplied by its base image.

## Delivery image

The R03 delivery artifact was assembled by appending one OCI application layer to the immutable prescribed AMD base. All eleven original base layers were preserved. Exact Linux x86_64 dependency wheels are exposed through `PYTHONPATH=/app/vendor`; the base ROCm Torch and torchvision were not replaced. The image downloads the pinned public model at first startup and contains no model weights or challenge fixtures. Its manifest, configuration and all layer availability were checked anonymously.

GPU timing and example-match figures below refer to the R03 source-path pilot. Python imports, full cold start and GPU inference of the final container image were not independently measured. The Dockerfile below is a source build route, separate from the OCI assembly used for delivery. The competition image reference is supplied in the submission form.

## Build and run on an AMD Linux host

The following build enables first-run download of the pinned public model. Docker build does not download model weights:

```sh
docker build --platform linux/amd64 --build-arg OCR_EMBED_MODEL=0 --build-arg OCR_ALLOW_MODEL_DOWNLOAD=1 -t figueira-ocr:r03 .
mkdir -p input output
docker volume create figueira-ocr-models
docker run -d --name figueira-ocr --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host -v figueira-ocr-models:/models -v "$PWD/input:/data:ro" -v "$PWD/output:/app/output" figueira-ocr:r03
```

Place an image in `input`. Wait for the worker to finish downloading and loading the model; inspect its log and health status before running the CLI:

```sh
docker logs figueira-ocr
docker inspect --format '{{.State.Health.Status}}' figueira-ocr
docker exec figueira-ocr python3 /app/healthcheck.py
docker exec figueira-ocr python3 /app/app.py --input-image /data/image.png
```

The result is `output/image_output.json`. The model volume is persistent across container replacements. Network access and enough storage for about 15.5 GiB of model weights are required on first run. Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The default build embeds weights and disables runtime downloads; the flags above deliberately select the runtime-download variant.

## Verified results and limits

R03 was tested through the source path on an AMD Radeon Pro W7900D, with PyTorch 2.13.0+rocm10.0.0. All ten published challenge examples matched after the permitted normalization; all ten CLI calls completed in under 30 seconds. Worker startup with weights already present was 26.17 s, the first CLI call 10.020 s, and the other calls 0.425–0.541 s. Peak total VRAM sampled every three seconds was 16.44 GiB. See [RESULTS.md](RESULTS.md).

These are source-path development measurements, not hidden grading results. The full container cold start with runtime model download has not been measured; do not apply the preloaded startup time to that path. Public examples do not establish general OCR accuracy. Official sample images, answer manifests, outputs, raw logs and private infrastructure identifiers are excluded from this repository.

## Software checks

Fifteen GPU-free contract, socket and worker lifecycle tests use invented strings and an injected fake model:

```sh
python3 -m unittest discover -s tests -v
```

The tests cover file output, normalization, formats, readiness, concurrent clients, duplicate-worker prevention, disconnect/timeout recovery and shutdown. They do not instantiate the real model or prove GPU inference.
