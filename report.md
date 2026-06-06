# Gunpla Studio Report

## Problem Statement

It's fun to build Gunpla (or Gundam robot models).
See: https://global.bandai-hobby.net/en-us/gunpla/ for examples of model kits.

Once built, it is fun to pose them and take pictures of them. Wouldn't it be great to take the models that you just built, pose them, and then put them on other backgrounds?

Gunpla Studio lets you do just that.

## Data Sources



## Related Work



## Review of Relevant Prior Work and Literature



## Evaluation Strategy & Metrics

The primary metric should be mask mean Intersection over Union (mIoU), because the application depends on the overlap between predicted and ground-truth Gunpla pixels. A model that finds the object but loses antennas, weapons, or feet will produce visibly poor composites, and mIoU directly penalizes those errors.

Secondary metrics should include Dice/F1 score, precision, recall, boundary F1, and detection confidence calibration. Dice is useful for small objects because it is sensitive to overlap quality. Precision measures background leakage into the mask. Recall measures missing robot parts. Boundary F1 is important because compositing quality depends heavily on edge accuracy.

## Metric Selection and Justification

mIoU is the critical metric because the task is segmentation, not only classification or detection. It evaluates the exact target surface area used for compositing. Dice/F1 complements mIoU by being easier to interpret for foreground/background imbalance. Precision and recall diagnose different failure modes: low precision means the app carries old background into the composite, while low recall means the robot appears clipped. Boundary F1 is included because a mask with high interior overlap can still look bad if edges are jagged or incomplete.

## Modeling Approach

The project uses a common `BaseSegmenter` interface with three implementations:

- Naive baseline: ???
- Classical ML model: ???
- Deep learning model: uses a fine-tuned YOLO11n-seg model for single-class Gunpla segmentation.

The interface allows the Gradio app to switch methods without changing UI logic.

## Data Processing Pipeline

1. Collect raw Gunpla images from multiple camera devices, lighting conditions, poses, and backgrounds.
2. Annotate each visible Gunpla with a polygon mask using an annotation tool.
3. Export labels in YOLO segmentation format with one class: `gunpla`.
4. Split data into train, validation, and test sets by physical model where possible, not just by image, to reduce leakage.
5. Build image summary features with `scripts/build_features.py` to ???
6. Train YOLO11n-seg using `scripts/train_model.py`.
7. Save best weights to `models/gunpla_yolo11n_seg.pt`.
8. Evaluate all models on the same held-out test set.

Each step supports reproducibility and lowers leakage risk. The split-by-model recommendation matters because near-duplicate photos of the same kit can overstate performance.

## Hyperparameter Tuning Strategy

Initial tuning should focus on image size, epochs, augmentation strength, confidence threshold, and mask post-processing. Recommended experiments:

- Image sizes: 512, 640, 768.
- Epochs: 30, 50, 100 with early stopping.
- Batch size: largest stable batch for available GPU memory.
- Confidence threshold: 0.15, 0.25, 0.40.
- Augmentations: compare default YOLO augmentations against stronger brightness, scale, perspective, and copy-paste settings.

The validation set should select hyperparameters using mIoU first, then boundary F1 and qualitative composite review.

## Models Evaluated

### Naive Baseline

The naive baseline predicts a centered ellipse. It is intentionally weak but useful because many product-style photos have centered objects. It establishes the minimum acceptable performance threshold.

### Classical ML Model

The classical model uses ???. It provides a data-free segmentation method that can outperform the naive baseline when the Gunpla is centered and visually separated from the background.

### Deep Learning Model

The deep learning model fine-tunes YOLO11n-seg. It is expected to perform best because it learns shape, pose, color, and context cues from annotated data and returns instance masks directly.

## Results

The table below is a planned reporting template. Scores should be filled after training and evaluating on the held-out test set.

| Model | mIoU | Dice/F1 | Precision | Recall | Boundary F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Naive baseline | TBD | TBD | TBD | TBD | TBD | Center prior only |
| Classical ML | TBD | TBD | TBD | TBD | TBD | Sensitive to background clutter |
| YOLO11n-seg fine-tuned | TBD | TBD | TBD | TBD | TBD | Expected best model |

Confusion matrices are less central for a single-class segmentation task than mask metrics, but pixel-level foreground/background confusion counts should be visualized as true foreground, false foreground, missed foreground, and true background.

## Visualizations and Confusion Matrices Where Appropriate

visualizations:

- Predicted mask overlaid on source image.
- Composite output on several background types.
- Pixel-level foreground/background confusion heatmap.
- Metric distributions by lighting condition, pose, model color, and background type.
- Precision-recall curve over confidence thresholds for the YOLO model.

## Error Analysis

Five specific misprediction categories to inspect:

1. Antennas or V-fins missing from the mask.
   Root cause: thin structures are underrepresented and hard to label consistently.
   Mitigation: add close-up examples, train at larger image size, and use boundary-aware review.

2. Weapons or backpacks excluded.
   Root cause: annotations may inconsistently include accessories.
   Mitigation: define annotation policy that all attached accessories are part of `gunpla`.

3. Background leakage around shadows.
   Root cause: shadows share object-adjacent color and texture.
   Mitigation: diversify lighting data and add shadow-heavy validation examples.

4. White Gunpla on white desk under-segmented.
   Root cause: low foreground/background contrast.
   Mitigation: collect low-contrast examples and tune augmentations for exposure variation.

5. Transparent effect parts removed or partially masked.
   Root cause: transparent plastic has weak visual boundaries.
   Mitigation: collect examples with clear parts and review whether transparent accessories should be included.

## Experiment Write-Up

Ideas: try different # of images... what's the minimum to get it working? or no fine tuning at all?

## Conclusions



## Future Work

more data?

## Commercial Viability Statement

This is not yet suitable for commercial use without more data, stronger evaluation, model monitoring, and rights review for backgrounds and training images. It could become commercially viable for hobbyist content creation if it demonstrates reliable masks across many kits, poses, lighting conditions, and user camera devices. Commercial deployment would also need privacy controls for uploaded photos and clear policies for copyrighted background assets.

## Ethics Statement

The project should use images collected with permission and avoid scraping copyrighted product photography without rights. Users should be informed that uploaded images may contain personal spaces or identifying metadata. Training and evaluation should include varied camera devices and lighting environments so the tool does not only work for ideal studio setups. The app should not imply official affiliation with Bandai, Sunrise, or the Gundam franchise unless such rights are obtained.
