# Model packages from armfarm01

Downloaded 2026-11-14 from the `hackathons` org (`6d6c7293-bc67-43e4-9c8b-31c76aac27d3`). The files are gitignored
(`model.bin` alone is 141 MB), so keep a backup somewhere other than this computer.

armfarm01's config asks for `latest`, but the machine had loaded **version 1** of each package (its mlmodel services
report `.../sortit-patchcore-1/model.xml` and `.../sortit-yolocls-1/model.onnx`). Use version 1 to reproduce the
machine; `LocalPatchCoreScorer` defaults to it.

| Package / version | Runs on armfarm01 as | Training data | Archive sha256 |
|---|---|---|---|
| `sortit-patchcore/1` | `ov` (viam:openvino:mlmodel, GPU, f32) behind `vision-1` | 61 normal / 12 anomalous | `c98835549c1065e2c78dfa7fbbdd076c17a2323641a6c66f7ede4540727458a2` |
| `sortit-patchcore/latest` | not loaded on the machine as of 2026-11-14 | 83 normal / 46 anomalous | `bd7233f2a879adffcff190ff5a4c1aade7730f3efcafb530e74f280839ab5347` |
| `sortit-yolocls/1` (= `latest`) | `mlmodel-2` (viam:openvino:mlmodel, NPU) behind `vision-2` | | `92bc9740862a554ed45d2d390d0ef81c399c4dce00a5cf73a2ab101ae36b574a` |

PatchCore: 256x256 input `input` `[1,3,256,256]` float 0-1; outputs `pred_score`, `anomaly_map`; anomalib 2.7.0.
yolocls: `model.onnx` with `labels.txt` (`anomalous`, `normal`).

Parity check (same input tensor, 2026-11-14): local v1 on the Mac CPU 0.5106, armfarm01 `ov` on the GPU 0.5106;
`vision-1` end to end 0.5173. Local `latest` gave 0.8228 on the same image, a different model.

Re-download (each lands as `models/<name>/<version>/<name>.tar.gz`; extract it in place with `tar xzf`):

```bash
viam packages export --org-id=6d6c7293-bc67-43e4-9c8b-31c76aac27d3 --name=sortit-patchcore --version=1 --type=ml_model --destination=models/sortit-patchcore
viam packages export --org-id=6d6c7293-bc67-43e4-9c8b-31c76aac27d3 --name=sortit-patchcore --version=latest --type=ml_model --destination=models/sortit-patchcore
viam packages export --org-id=6d6c7293-bc67-43e4-9c8b-31c76aac27d3 --name=sortit-yolocls --version=1 --type=ml_model --destination=models/sortit-yolocls
```
