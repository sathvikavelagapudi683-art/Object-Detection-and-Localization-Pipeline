# Scope, provenance and limitations

## What the project description supports

The project résumé describes WildWatch as an image/video inference pipeline using pretrained YOLO11n. It reports exporting predicted wildlife-related class labels, bounding boxes and confidence scores to CSV, testing multi-object scenes, and documenting classification/localisation limitations. It explicitly states that the model was used for inference rather than trained from scratch.

## What this repository contains

This is a clean, reproducible reference implementation of that described inference workflow. It loads an Ultralytics model (default `yolo11n.pt`), runs prediction on a single image or video, saves annotated media, and exports one CSV row per detection. It does not claim to be the original project script.

## What is not included

- Original source files or notebook: not located in the available project files.
- Original input images/videos and manually curated evaluation set: not located.
- Original trained weights: not located; the documented project used pretrained YOLO11n.
- Detection metrics such as precision, recall, mAP, IoU, or per-class accuracy: no values were present in the available project description, so none are reported here.

## Interpretation

YOLO confidence scores are model scores, not calibrated probabilities. A bounding box describes the model's estimated image region; it does not establish perfect object boundaries. Multi-object scenes can expose class confusion, missed objects, overlapping boxes, and imprecise localization. The résumé notes class-confusion/localization limitations but does not supply a quantitative evaluation dataset or metric values.

## Reproducibility

The implementation downloads/loads the pretrained checkpoint through the Ultralytics package when first used. Results can vary with checkpoint version, input media, confidence threshold, image size, and runtime/device. Preserve input media and output CSV separately if you need to repeat a particular run. Do not treat this repository as a trained wildlife-specific detector unless a separately validated wildlife checkpoint is supplied.
