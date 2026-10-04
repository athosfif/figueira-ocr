# R03 GPU source-path results

Measured on October 3, 2026 (Brasilia), using AMD Radeon Pro W7900D, Qwen2.5-VL-7B-Instruct FP16 and PyTorch 2.13.0+rocm10.0.0. Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

| Measurement | Observed result |
| --- | --- |
| Published-example files | 10 |
| Matches after permitted normalization | 10/10 |
| JSON output contract | 10/10 valid |
| CLI calls below 30 seconds | 10/10 |
| Worker startup, prepared weights | 26.17 s |
| First CLI call | 10.020 s |
| Subsequent CLI calls | 0.425–0.541 s |
| Ten calls in total | 14.31 s |
| Peak total VRAM, sampled every 3 s | 16.44 GiB |
| GPU-free software tests | 15/15 locally and remotely |

The same worker served all calls, remained alive afterwards and shut down cleanly. The ten images were raster examples from the published challenge PDF, re-encoded as PNG, JPEG or TIFF. This is not hidden evaluation and does not establish general accuracy. Source-path timing does not measure a full container cold start or the runtime model download. A sampled VRAM maximum can miss peaks between samples. No expected example strings or private logs are included here.
