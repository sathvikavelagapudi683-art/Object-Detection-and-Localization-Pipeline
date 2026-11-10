# WildWatch — Object Detection & Localization Pipeline

**Stack:** pretrained YOLO11n, Ultralytics, OpenCV, Python  
**Task:** image/video inference, multi-object detection, bounding-box localization, and CSV export.

WildWatch is described in the project résumé as an inference pipeline using pretrained YOLO11n to identify wildlife-related object classes. It extracts predicted labels, bounding boxes, and confidence scores, and exports detections to CSV. Multi-object scenes were tested and class-confusion/localization limitations were documented. The model was used for inference, not trained from scratch.

> **Repository provenance:** The original project script, sample media, and checkpoint were not found in the available project files. This repository therefore contains a clean, reproducible reference implementation of the documented inference workflow—not a claim that the original files were recovered. No unverified detection metrics are reported.

## Features

- Run pretrained YOLO11n inference on one image or one video.
- Save annotated image or MP4 output.
- Export each detection to CSV: source, frame index, class ID/name, confidence, and pixel-coordinate bounding box (`x1`, `y1`, `x2`, `y2`).
- Configure confidence threshold and optional inference device.
- Unit tests for confidence validation, bounding-box validation, and CSV export.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run on an image

```bash
python -m wildwatch.inference \\
  --input path/to/wildlife_image.jpg \\
  --output-dir runs/wildwatch \\
  --model yolo11n.pt \\
  --confidence 0.25
```

## Run on a video

```bash
python -m wildwatch.inference \\
  --input path/to/wildlife_video.mp4 \\
  --output-dir runs/wildwatch \\
  --model yolo11n.pt \\
  --confidence 0.25
```

The first run may download the pretrained YOLO11n checkpoint if it is not already available in the Ultralytics cache. Internet access may be required for that initial download. Use `--device cpu` to force CPU inference or `--device 0` for a CUDA device index supported by the installed PyTorch runtime.

## Outputs

For an input named `wildlife_image.jpg`, the output directory contains:

- `wildlife_image_detections.jpg` — annotated image.
- `wildlife_image_detections.csv` — one row per detection.

For `wildlife_video.mp4`, it contains an annotated MP4 and a CSV with frame indices. Empty detections produce a CSV with headers and no data rows.

## Tests

```bash
python -m unittest discover -s tests -v
```

The included tests do not download model weights and do not evaluate detection quality. They validate utility functions and CSV serialization only.

## Limitations

This is a pretrained general-purpose detector workflow, not a wildlife-specific model trained by this project. Detection quality depends on the classes supported by the selected pretrained checkpoint, the scene, camera quality, object size/occlusion, confidence threshold, and model version. The available project description notes classification/localization errors in mixed-object scenes, but provides no numeric precision, recall, mAP, or IoU results. See [`docs/SCOPE_AND_LIMITATIONS.md`](docs/SCOPE_AND_LIMITATIONS.md).
