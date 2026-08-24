<h1 align="center">Context-structured Video Anomaly Detection with Large Vision-Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2607.19077"><img src="https://img.shields.io/badge/ArXiv-2607.19077-red" alt="ArXiv"></a>
  <a href="https://dj991108.github.io/CSI-VAD/"><img src="https://img.shields.io/badge/Project-Page-Blue" alt="Project Page"></a>
</p>

This is the official repository for "Context-structured Video Anomaly Detection with Large Vision-Language Models," published at AVSS 2026. This repository contains inference code for context-structured video anomaly detection on a single input video.

## Installation

```bash
git clone https://github.com/dj991108/CSI-VAD.git
cd CSI-VAD

conda env create -f environment.yml
conda activate csi-vad
```

Download the pinned Hugging Face snapshots and the detector/tracker weights:

```bash
python scripts/download_models.py
python scripts/verify_environment.py
```

Model downloads are stored in the Hugging Face cache and the ignored local `weights/` directory. Review and accept the licenses of all third-party packages and model weights before downloading them.

## Inference

Run inference with the default frame budget:

```bash
python infer.py --video /path/to/video.mp4 --output outputs/result.json
```

Use `--num-frames` to configure the frame budget.

The default final score is Noisy-OR and the default final label uses OR voting. All supported aggregation methods can be selected from the command line:

```bash
python infer.py \
  --video /path/to/video.mp4 \
  --output outputs/result.json \
  --score-aggregation mean \
  --label-aggregation majority
```

Validate paths and inspect the effective configuration without loading models:

```bash
python infer.py \
  --video /path/to/video.mp4 \
  --output outputs/result.json \
  --dry-run
```

Use `python infer.py --help` for all options. Interrupted runs reuse the ignored `.csi_vad_work/` artifacts. Pass `--clean-workdir` to rebuild them.

## Output

`result.json` contains:

- label, score, and explanation from the environment, object, and time branches;
- Max, Mean, and Noisy-OR score aggregations;
- OR, majority, and AND label aggregations;
- the selected final prediction, effective configuration, model revisions, video metadata, and stage timings.

Generated frames, object overlays, captions, and context files remain in the ignored work directory and are not copied into the final result.

### Context branch results

The final result contains separate predictions from the environment, object, and time branches. Inspect them with:

```bash
python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("outputs/result.json").read_text(encoding="utf-8"))

for name in ("environment", "object", "time"):
    branch = result["branches"][name]
    print(f"\n[{name}]")
    print(f"label: {branch['label']}")
    print(f"score: {branch['score']:.4f}")
    print(f"explanation: {branch['explanation']}")
PY
```

## Citation

```bibtex
@inproceedings{kim2026csivad,
  title     = {Context-structured Video Anomaly Detection with Large Vision-Language Models},
  author    = {Kim, Dongjun and Oh, Changjae and Cavallaro, Andrea and Mo, Jeonghoon},
  booktitle = {IEEE International Conference on Advanced Video and Signal Based Surveillance},
  year      = {2026}
}
```

## License

Original CSI-VAD source code is released under the [MIT License](LICENSE). Third-party packages and model weights retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the object branch uses BoxMOT/StrongSORT and is subject to their copyleft license terms.
