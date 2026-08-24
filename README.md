<h1 align="center">Context-structured Video Anomaly Detection with Large Vision-Language Models</h1>

<p align="center">Official inference code for AVSS 2026.</p>

<p align="center">
  <a href="https://dj991108.github.io/CSI-VAD/"><img src="https://img.shields.io/badge/Project%20Page-CSI--VAD-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2607.19077"><img src="https://img.shields.io/badge/arXiv-2607.19077-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

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

Run the default 32-frame setting:

```bash
python infer.py --video /path/to/video.mp4 --output outputs/result.json
```

Set a custom frame budget with `--num-frames`:

```bash
python infer.py \
  --video /path/to/video.mp4 \
  --output outputs/result_256.json \
  --num-frames 256
```

The default final score is Noisy-OR and the default final label uses OR voting. All paper aggregation variants can be selected from the command line:

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

Each context branch can be inspected separately after inference:

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

## Tests

```bash
pytest -q
python -m compileall -q csi_vad scripts infer.py
```

The automated suite is CPU-only. Full inference additionally requires the documented GPU environment and downloaded weights.

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

Original CSI-VAD source code is released under the [MIT License](LICENSE). Third-party packages and model weights retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the paper-faithful object branch uses BoxMOT/StrongSORT and is subject to their copyleft license terms.
