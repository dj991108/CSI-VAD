# Third-Party Notices

The repository-level MIT License applies only to original CSI-VAD source code. The following dependencies, their transitive dependencies, and all model weights are **not covered by the CSI-VAD MIT License**. They remain subject to their respective licenses and terms.

| Component | Use in CSI-VAD | Upstream license |
| --- | --- | --- |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) | StrongSORT runtime integration | AGPL-3.0 |
| [StrongSORT](https://github.com/dyhBUPT/StrongSORT) | Paper object tracker | GPL-3.0 |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Object detection | Core code and the downloaded RF-DETR Medium COCO checkpoint are Apache-2.0. The installed package also contains optional Roboflow Platform integration files under the Platform Model License 1.0; CSI-VAD does not call that integration. |
| [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Environment, object, and caption inference | See the model card and repository license |
| [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Temporal summarization and inference | See the model card and repository license |
| [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | Caption embeddings | Apache-2.0 |
| [OpenCV](https://opencv.org/) | Video decoding and rendering | Apache-2.0 |

CSI-VAD does not vendor BoxMOT, StrongSORT, RF-DETR, or model weights. The setup scripts download or install them from their upstream distribution locations. Using the complete runtime may create obligations beyond those of the MIT-licensed CSI-VAD files. Review the upstream terms and obtain institutional legal guidance when required.
