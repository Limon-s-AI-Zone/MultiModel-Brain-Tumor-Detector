# Brain Tumor Detection in MRI — CV and portfolio text

Repository: https://github.com/mksl49/brain-tumor-detection-MRI
Period: January – February 2026

## GitHub "About" line

Comparative study of YOLOv8, Faster R-CNN, and DETR for brain tumor detection in MRI — unified data pipeline, COCO evaluation, and multi-model inference.

Topics: `object-detection` `medical-imaging` `mri` `brain-tumor` `yolov8` `faster-rcnn` `detr` `detectron2` `pytorch` `computer-vision`

## CV entry (PhD application)

**Brain Tumor Detection in MRI: A Comparison of One-Stage, Two-Stage, and Transformer Detectors** | PyTorch, Ultralytics, Detectron2, Hugging Face Transformers | Jan – Feb 2026 | [GitHub](https://github.com/mksl49/brain-tumor-detection-MRI)

- Benchmarked YOLOv8m, Faster R-CNN (ResNet-101-FPN), and DETR (ResNet-50) for four-class tumor localization on 5,249 annotated MRI slices under a shared data pipeline and COCO evaluation protocol.
- YOLOv8m reached 0.781 mAP@0.5:0.95 and 0.963 mAP@0.5 (F1 0.946), compared with 0.562 for Faster R-CNN and 0.508 for DETR; analysis by IoU threshold and object size showed that the two weaker models detect tumors but localize small lesions poorly (small-object mAP ≈ 0.26).
- Adapted each detector to the domain with anatomy-preserving augmentation, CLAHE, extended anchor scales, and class-imbalance handling (repeat-factor sampling, class-weighted loss); built a single pipeline for data conversion, training, evaluation, and multi-model inference.

### Two-bullet version

- Compared YOLOv8m, Faster R-CNN (R101-FPN), and DETR (R50) for four-class brain tumor detection on 5,249 MRI slices under a unified data and COCO evaluation pipeline.
- YOLOv8m achieved 0.781 mAP@0.5:0.95 and 0.963 mAP@0.5, against 0.562 and 0.508 for Faster R-CNN and DETR; identified small-lesion localization as the main failure mode of the latter two.

### One-line version

Benchmarked YOLOv8, Faster R-CNN, and DETR for brain tumor detection in MRI (5,249 slices, 4 classes); best model reached 0.781 mAP@0.5:0.95 and 0.963 mAP@0.5.

## Portfolio

**Title:** Brain Tumor Detection in MRI

**Tagline:** Three detector families, one dataset, one evaluation protocol — which one actually works for tumor localization?

**Summary**

This project compares a one-stage CNN detector (YOLOv8m), a two-stage CNN detector (Faster R-CNN with a ResNet-101-FPN backbone), and a transformer detector (DETR with ResNet-50) on the task of locating and classifying brain tumors — glioma, meningioma, pituitary, and no-tumor — in MRI slices. I wrote a data pipeline that converts the Kaggle dataset from per-class YOLO folders into both a merged YOLO dataset and COCO JSON, so all three frameworks train and evaluate on exactly the same 4,737 training and 512 validation images.

Each model was adapted to MRI rather than run with defaults: augmentations avoid vertical flips and perspective warps that would break anatomy, the DETR pipeline uses CLAHE contrast enhancement and a class-weighted loss, and Faster R-CNN uses extended anchor sizes and repeat-factor sampling for the under-represented class.

YOLOv8m was the clear winner at 0.781 mAP@0.5:0.95. The more interesting result is in the breakdown: Faster R-CNN is nearly as good as YOLOv8 at IoU 0.5 (0.927 vs. 0.963) but drops sharply at IoU 0.75 (0.592 vs. 0.897), meaning it finds the tumors but draws loose boxes. Both Faster R-CNN and DETR score around 0.26 mAP on small objects.

**Results**

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|
| YOLOv8m | 0.781 | 0.963 | 0.897 |
| Faster R-CNN (R101-FPN) | 0.562 | 0.927 | 0.592 |
| DETR (R50) | 0.508 | 0.762 | 0.573 |

**Highlights**

- Unified YOLO → COCO data pipeline so three frameworks share identical data.
- Domain-specific augmentation and class-imbalance handling for each model.
- COCO evaluation with breakdown by IoU threshold and object size.
- One command to run data preparation, EDA, training, evaluation, and comparison; one script to run all three models on a new image and save side-by-side predictions.
- Experiments tracked with Weights & Biases; trained on an NVIDIA RTX A5000.

**Tech:** Python, PyTorch, Ultralytics YOLOv8, Detectron2, Hugging Face Transformers, PyTorch Lightning, Albumentations, pycocotools, Weights & Biases

**Suggested images:** `results/model_comparison.png`, `predictions/Tr-gl_1295_{yolo,faster_rcnn,detr}.png`

**Next steps:** held-out test set and multi-seed runs, longer DETR schedule, Deformable DETR / RT-DETR, per-class error analysis.
