FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ARG OCR_ALLOW_MODEL_DOWNLOAD=0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_MODEL_DIR=/models/qwen \
    OCR_STARTUP_TIMEOUT=570 \
    OCR_INFERENCE_TIMEOUT=30 \
    OCR_ALLOW_MODEL_DOWNLOAD=${OCR_ALLOW_MODEL_DOWNLOAD} \
    OCR_MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct \
    OCR_MODEL_REVISION=cc594898137f460bfe9f0759e9844b3ce807cfb5

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Freeze the ROCm torch packages supplied by the mandated base image.
RUN python3 -c "import json, torch; from importlib.metadata import version; from pathlib import Path; v={n:version(n) for n in ('torch','torchvision')}; assert '+rocm' in v['torch'] and torch.version.hip, v; v['torch_file']=str(Path(torch.__file__).resolve()); Path('/tmp/rocm-base-versions.json').write_text(json.dumps(v)); Path('/tmp/rocm-constraints.txt').write_text(''.join(n+'==='+v[n]+'\n' for n in ('torch','torchvision')))" \
    && python3 -m pip install --no-cache-dir --only-binary=:all: -c /tmp/rocm-constraints.txt -r /app/requirements.txt \
    && python3 -m pip check \
    && python3 -c "import json, torch; from pathlib import Path; from importlib.metadata import version; v=json.loads(Path('/tmp/rocm-base-versions.json').read_text()); assert all(version(n)==v[n] for n in ('torch','torchvision')), v; assert str(Path(torch.__file__).resolve())==v['torch_file'], 'ROCm torch path changed'; assert torch.version.hip, 'ROCm torch required'"

COPY app.py backends.py ocr_core.py worker.py healthcheck.py /app/
COPY scripts/fetch_model.py /app/scripts/fetch_model.py

ARG OCR_EMBED_MODEL=1
RUN if [ "$OCR_EMBED_MODEL" = "1" ]; then python3 /app/scripts/fetch_model.py \
    --repo "$OCR_MODEL_REPO" \
    --revision "$OCR_MODEL_REVISION" \
    --destination "$OCR_MODEL_DIR"; \
    elif [ "$OCR_EMBED_MODEL" != "0" ]; then exit 1; fi

RUN mkdir -p /app/output && chmod 0777 /app/output
HEALTHCHECK --interval=2s --timeout=3s --start-period=600s --retries=3 CMD ["python3", "/app/healthcheck.py"]
CMD ["python3", "/app/worker.py"]
