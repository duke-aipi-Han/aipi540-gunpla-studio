# Gunpla Studio Experiments

## Summary

This log now tracks only runs trained after the image-orientation issue was fixed. Older runs were discarded because they used invalid training data.

The two fixed-data runs used the same training setup except for the YOLO model size:

- data: `data/processed/gunpla-yolov7/data.yaml`
- image size: `768`
- batch size: `2`
- augmentation: `light`
- mask ratio: `4`
- epochs requested: `100`
- device: GPU `0`

On the Ultralytics validation split, `yolo11n-seg` is currently the better deployment candidate. It slightly outperformed `yolo11s-seg` on mask precision, mask mAP50, mask mAP50-95, box mAP50-95, and validation segmentation loss while producing a much smaller model file.

The main metric for this project is mask quality, especially mask precision, mask recall, mask mAP50, and mask mAP50-95. Box metrics are secondary because the app needs pixel-level foreground masks for background replacement.

## Run Comparison

| Run | Model | Data | Augmentation | Epochs Run | Best Epoch | Box mAP50 | Box mAP50-95 | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 | Val Seg Loss | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gunpla_yolo11n_seg_orientation_fixed` | `yolo11n-seg.pt` | fixed `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 62 | 60 | 0.995 | 0.782 | 0.998 | 1.000 | 0.995 | 0.894 | 0.538 | Best validation result so far. Smaller model and slightly stronger masks than `11s`. |
| `gunpla_yolo11s_seg_orientation_fixed` | `yolo11s-seg.pt` | fixed `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 66 | 49 | 0.993 | 0.776 | 0.970 | 1.000 | 0.993 | 0.870 | 0.716 | Strong result, but did not beat `11n` on validation despite being larger. |

## Interpretation

The fixed-orientation data changed the training outcome substantially. Both models now score very highly on the validation split, which suggests the previous orientation problem was a major source of poor training signal.

`yolo11s-seg` did not provide a validation benefit over `yolo11n-seg` in this run. The larger model reached perfect mask recall, but so did `11n`; meanwhile `11n` had better mask precision, better strict mask mAP, and lower validation segmentation loss.

For the Gradio app, the current validation result favors `gunpla_yolo11n_seg_orientation_fixed.pt`. Before final deployment, compare both fixed-orientation models on the notebook test set after applying EXIF orientation consistently and clearing stale output images.

