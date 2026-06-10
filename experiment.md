# Gunpla Studio Experiments

## Summary

The best overall run so far is `yolo11n-seg` with light augmentation at `imgsz=768`, batch size `2`, and the default `mask_ratio=4`. A later `mask_ratio=2` run improved mask mAP50, precision, and recall, but reduced mask mAP50-95. For compositing, `mask_ratio=2` is worth visual review, but `mask_ratio=4` remains the cleaner metric winner on stricter IoU.

The main metric for this project is mask quality, especially `metrics/mAP50(M)`, `metrics/mAP50-95(M)`, mask precision, and mask recall. Box metrics are secondary because the application needs pixel-level foreground masks for compositing.

## Run Comparison

| Run | Model | Data | Augmentation | Epochs Run | Best Epoch | Box mAP50 | Box mAP50-95 | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gunpla_yolo11n_seg` | `yolo11s-seg.pt` | `gunpla-yolov7 | strong: mosaic `0.7`, mixup `0.08`, copy-paste `0.25`, geometric transforms | 27 | 12 | 0.214 | 0.054 | 0.122 | 0.304 | 0.094 | 0.019 | Poor run. Strong augmentation destabilized training. |
| `gunpla_yolo11s_seg_img768_b2_noaug_20260607_225002` | `yolo11s-seg.pt` | `gunpla-yolov7 | intended no aug, but Ultralytics default mosaic was still active: mosaic `1.0` | 100 | 61 | 0.594 | 0.226 | 0.725 | 0.217 | 0.252 | 0.136 | High precision, weak recall. Not actually no augmentation. |
| `gunpla_yolo11n_seg_img768_b2_none_20260607_232159` | `yolo11n-seg.pt` | `gunpla-yolov7` | none: mosaic `0`, flips off, geometric/color aug off | 37 | 17 | 0.650 | 0.201 | 0.381 | 0.339 | 0.269 | 0.074 | Better recall behavior, but lower mask mAP50-95. Early stopped. |
| `gunpla_yolo11n_seg_img768_b2_light_20260608_072123` | `yolo11n-seg.pt` | `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 75 | 60 | 0.896 | 0.423 | 0.602 | 0.600 | 0.553 | 0.258 | Best run so far. Balanced precision/recall and strongest mask mAP. |
| `gunpla_yolo11s_seg_img768_b2_light_20260608_075636` | `yolo11s-seg.pt` | `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 100 | 88 | 0.963 | 0.440 | 0.535 | 0.575 | 0.473 | 0.164 | Better box detector than `11n`, but worse masks. Not preferred for compositing. |
| `gunpla_yolo11n_seg_img768_b2_light_20260608_082711` | `yolo11n-seg.pt` | `gunpla-yolov7` | light + finer masks: same light aug, `mask_ratio=2` | 87 | 72 | 0.985 | 0.478 | 0.711 | 0.700 | 0.592 | 0.222 | Stronger mask mAP50, precision, and recall, but lower mask mAP50-95 than `mask_ratio=4`. |

## Interpretation

Light augmentation is clearly better than no augmentation and much better than strong augmentation. The model benefits from modest color/geometric variation, but heavy mosaic, mixup, and copy-paste appear to damage mask learning on this small dataset.

`yolo11n-seg` is currently outperforming `yolo11s-seg` for the segmentation objective. The `11s + light` run improved box detection, but mask mAP50 and mask mAP50-95 were both lower than `11n + light`. Since the app depends on mask quality for background replacement, `11n + light` is the better current model despite being smaller.

The `mask_ratio=2` run is a mixed result. It improved detection and loose mask overlap:

```text
mask_ratio=4: mask precision 0.602, recall 0.600, mAP50 0.553, mAP50-95 0.258
mask_ratio=2: mask precision 0.711, recall 0.700, mAP50 0.592, mAP50-95 0.222
```

Visually, `mask_ratio=2` finds more of the object and produces higher-confidence predictions, but it also shows more mask spill/background blobs in some validation examples. `mask_ratio=4` is more conservative and scores better on stricter IoU. For the app, compare both on real compositing examples before choosing the deployment model.

