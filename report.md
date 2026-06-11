# Gunpla Studio Report

## Problem Statement

It's fun to build Gunpla (or Gundam robot models). See: [Bandai](https://global.bandai-hobby.net/en-us/gunpla/) for examples of model kits.

Once built, it is fun to pose them and take pictures of them. Wouldn't it be great to take the models that you just built, pose them, and then put them on custom backgrounds? This way you can share your latest build or pose on a background other than your cluttered home. 

![Example of Gundam](./report-images/gundam.jpg "Gundam")

Gunpla Studio lets you do just that. It's basically green-screen for your Gunpla!
Access it at: https://hw391-gunpla-studio.hf.space/

## Data Sources
While there are plenty of Gundam/Gunpla pictures on the internet, many of them are copyrighted. There are also no publicly available labeled (and segmented) Gunpla pictures either (at least not "hundreds").

So I took 200 pictures of my son's Gunpla creations with my phone, uploaded them into Roboflow's Annotate, and used their SAM3 autodetect feature to label those pictures to use as starting point for training using transfer learning on top of YOLO foundation model for instance segmentation.

## Related Work
There are plenty of hobby/community sites that have pictures of Gunpla/Gundams. Especially from the creators: [Bandai](https://global.bandai-hobby.net/en-us/gunpla/)

There are also lots of community sites such as [Gunpla Gallery](https://gunplagallery.com/builds). However, none of those images are labeled nor annotated. And they may be copyrighted.

There are very few public projects for Gunpla detection that I can find.

## Review of Relevant Prior Work and Literature
There are very few CV projects around gunpla detection. There is a tiny dataset with a trained model here: https://universe.roboflow.com/hotd-2/gunpla/dataset/1

Related are humanoid robot detection models, and the "robot" detection for some models such as SAM3 works pretty well. However, Gunpla models have many models with similar designs and color schemes, and deserve its own detection and segmentation model optimized for this use case.

The novelty is the dataset, as there were none openly available *and labeled* for segmentation (except for a handful of labeled images). This is now the biggest labeled Gunpla dataset for segmentation.

## Evaluation Strategy & Metrics

The goal of this project is to use instance segmentation to create a mask, so that the Gunpla can then be overlaid on top of a background image "seamlessly" so that it is visually pleasing.

The primary metric should be mask mean Intersection over Union (mIoU), because the application depends on the overlap between predicted and ground-truth Gunpla pixels. A model that finds the object but loses antennas, weapons, or feet will produce visibly poor composites, and mIoU directly penalizes those errors. Basically, the ideal mask is the training annotation, and want to match it as much as possible.

Secondary metrics should include Dice/F1 score, precision, recall, boundary F1, and detection confidence calibration. Dice is useful for small objects because it is sensitive to overlap quality. Precision measures background leakage into the mask. Recall measures missing robot parts. Boundary F1 is important because compositing quality depends heavily on edge accuracy.

## Metric Selection and Justification

mIoU is the critical metric because the task is segmentation for creating image masks, not only classification or detection. It evaluates the exact target surface area used for composite imaging. 

Dice/F1 complements mIoU by being easier to interpret for foreground/background imbalance. Precision and recall diagnose different failure modes: low precision means the app carries old background into the composite, while low recall means the robot appears clipped or missing parts. 

Boundary F1 is included because a mask with high interior overlap can still look bad if edges are jagged or incomplete.

## Modeling Approach

The project uses a common `BaseSegmenter` interface with three implementations:

- Naive baseline: A centered ellipse as a reasonable first attempt
- Classical ML model: GrabCut as part of openCV
- Deep learning model: uses a fine-tuned YOLO11n-seg model for single-class Gunpla segmentation.

The interface allows the Gradio app to switch methods without changing UI logic, hidden in the "Power user options", to compare results.

## Data Processing Pipeline

1. Collect raw Gunpla images from "clean" background, different Gundams, different poses, and different orientation.
2. Annotate each visible Gunpla with a polygon mask using an annotation tool (Roboflow Annotate)
3. Export labels in YOLO segmentation format with one class: `gunpla`.
4. Split data into train, validation, and test sets by physical model where possible, not just by image, to reduce leakage.
5. Train YOLO11n-seg using `scripts/train_model.py`.
6. Save best weights to `models/gunpla_yolo11n_seg.pt`.
7. Evaluate all models on the same held-out test set.

## Hyperparameter Tuning Strategy

Initial tuning should focus on image size, epochs, augmentation strength, confidence threshold, and mask post-processing. Recommended experiments:

- Image sizes: 512, 640, 768. 768 was settled on.
- Epochs: 30, 50, 100 with early stopping. 100 was the typical limit
The biggest tuning turned out to be Augmentations
- Augmentations: compare default YOLO augmentations against stronger brightness, scale, perspective, and copy-paste settings. Too much augmentation actually generated bad results for 11s.

The validation set should select hyperparameter using mIoU first, then boundary F1 and qualitative composite review.

## Models Evaluated
3 models were evaluated:
1) a Naive Baseline - using a centered oval where subject foregrounds are typical in environmental portraits
2) Classical ML Model - using GrabCut
3) 2 versions of the YOLO-seg model. Were limited to YOLO compatible because the annotations were exported in that format.

### Naive Baseline

The naive baseline predicts a centered ellipse. It is intentionally weak but useful because many product-style photos have centered objects. It establishes the minimum acceptable performance threshold.

### Classical ML Model

The classical model uses GrabCut (from OpenCV library) It provides a data-free segmentation method that can outperform the naive baseline when the Gunpla is centered and visually separated from the background. It generated decent results but often misses "parts".

### Deep Learning Model

The deep learning model fine-tunes YOLO11n-seg (nano). It is expected to perform best because it learns shape, pose, color, and context cues from annotated data and returns instance masks directly. This is balanced with available resources and CPU inference on host.

Experiments were run with YOLO11s (small - a 3-4x larger model than nano) to see if a larger model improves performance at cost of longer training and inference.

## Results

This section shows the resulting metric tables and discuss the interpretation of the differences.

Note: The test set was "difficult" because it also included some real-world photos and clutter, so performance is worse than validation.

| model | mIoU |	Dice/F1 | Precision | Recall | Boundary F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Naive baseline | 0.3602 | 0.5237 | 0.3895 | 0.8766 | 0.0199 | Center Oval |
| Classical ML (GrabCut) | 0.5124 | 0.6475 | 0.5408 | 0.8978 | 0.4042 | better and good edge detection |
| YOLO11n-seg final | 0.5862 | 0.6935 | 0.7259 | 0.6805 | 0.4236 | best model after some tuning |

![Result Bar Chart](./report-images/result-barchart.png)

The YOLO11n-seg model beats all the other models on this difficult test set, and generally have good real-world results (but not perfect). 

## Visualizations and Confusion Matrices Where Appropriate

visualizations:

Here are some validation results during training:
Ground Truth Labels:
![Labels](./report-images/val_batch1_labels.jpg)
Model Predictions:
![Predictions](./report-images/val_batch1_pred.jpg)

## Error Analysis

Five specific mis-prediction categories to inspect:

1. Weapons or wings/attachments excluded.
   Root cause: annotations may inconsistently include/exclude accessories.
   Mitigation: define annotation policy that all attached accessories are part of `gunpla`.
   Example: the long weapon was clipped
![Weapons clipped](./report-images/yolo11n-seg-error-3.png)

2. Strange artifacts
   Root cause: Likely caused by training data clutter or occlusions
   Mitigation: Cleaner training data
   Example:In the previous image, there are some weird artifacts on the floor.

3. Missing body parts
   Root cause: training data occlusions
   Mitigation: cleaner training data
   Example: in the previous image, the right lower leg is missing from mask.

4. Background leakage
   Root cause: Background surrounded by Gundam parts show up and not excluded
   Mitigation: Check annotation to make sure they were not included by mistake, as well as add more examples of this.
   Example: the area between the arm/weapon and body were included incorrectly
![Weapons clipped](./report-images/yolo11n-seg-error-1.png)

5. Straight clipping of body parts
   Root cause: Training data bounding boxes
   Mitigation: check training data to make sure it includes wings
   Example: The wings are clipped in a straight line, almost as if there's an invisible bounding box
![Weapons clipped](./report-images/yolo11n-seg-error-2.png)

## Experiment Write-Up

There were 1 purposeful experiment, and 1 accidental (aside from parameter tuning)

The main experiment was to compare YOLO11n (nano) vs YOLO11s (small). Nano is the smallest model (with weights coming out to 6MB), with small the next step up that comes out to 20MB.

To run this experiment, 11n was tuned to a good performance level (as shown above), and the exact same parameters were run to train 11s to isolate the effect.

Training/val comparison:

| Run | Model | Data | Augmentation | Epochs Run | Best Epoch | Box mAP50 | Box mAP50-95 | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 | Val Seg Loss | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gunpla_yolo11n_seg_orientation_fixed` | `yolo11n-seg.pt` | fixed `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 62 | 60 | 0.995 | 0.782 | 0.998 | 1.000 | 0.995 | 0.894 | 0.538 | Best validation result so far. Smaller model and slightly stronger masks than `11s`. |
| `gunpla_yolo11s_seg_orientation_fixed` | `yolo11s-seg.pt` | fixed `gunpla-yolov7` | light: mosaic `0.15`, degrees `5`, translate `0.06`, scale `0.2`, horizontal flip, erasing `0.05` | 66 | 49 | 0.993 | 0.776 | 0.970 | 1.000 | 0.993 | 0.870 | 0.716 | Strong result, but did not beat `11n` on validation despite being larger. |

* They both stopped short, converging at around 60 epochs (set to 100 with early stop)
* through all the metrics, the results were very close, and comparing best runs, but nano still beat small (marginally)
* However, 11n won the race because of its small size and faster, for same performance.

More experiments that was accidental were:
* too much augmentation. I used aggressive augmentation to start, and 11s actually stopped early because it was so bad. Reduced the augmentation parameters so that the model converged.
* Orientation - this was more of a mistake. For some reason, the images didn't have orientation, but the mask did, when I downloaded the annotations. Was getting really bad results until visual inspection of image-mask revealed the error and the simple fix. Also some of the annotations came back with overlapping instances (where it was just 1), so a script called clean_overlapping_masks were run to remove extra regions in the annotation to simplify training.

## Conclusions
For a "small" model, it performed surprisingly well (almost magical) on "clean" pictures. It was able to detect the Gunpla and combine it with the backgrounds pretty well. It was exciting to see the "green-screen" effect on something we built together. The results generally beat the original picture in terms of "fun" and "immersion". 

The classical ML model also worked surprisingly well (cleanly detecting the edges. However, it can't tell foreground from background, and makes major mistakes). This shows the limitation of manual "feature engineering" and why larger NN models that can discover features automatically is superior.

## Future Work
Here are a few ideas that ran out of time on, that could be done in the future:
1) More data, this dataset included a specific bias of gundams that were "cool" (which to my son means big wings and weapons and random attachments). Turned out coolness means too many edges and attachments to detect. Perhaps take pictures without attachments so it can learn what the base Gundam looks like. There are also other models in the Gundam universe that were neglected (the "bad guys"). Also, only used 1 camera for majority of pictures (my iphone)
2) more augmentation - find a good balance between "lite" and "too much". It's probably a little light right now because of major mistakes on the real-world examples (like upside-down). But due to lack of training progress on aggressive augmentation, I stayed with light.
3) Some transformations - I noticed that the lighting between the model and the background doesn't match, so maybe some hue/color image processing to at least match the color will look better.

## Commercial Viability Statement

This is not yet suitable for commercial use without more data, stronger evaluation, and model monitoring. It could become viable for hobbyist content creation if it demonstrates reliable masks across many kits, poses, lighting conditions, and user camera devices. It took a lot of effort just to collect 200 images.

Commercial deployment would also need privacy controls for uploaded photos and clear policies for copyrighted background assets. Currently, nothing is saved to avoid this issue.

## Ethics Statement

The project used images collected personally, with a few images from hobbyist pages (so presumed to be common). So no major copyright and IP concerns.

Users should be informed that uploaded images may contain personal spaces or identifying metadata (EXIF photo information). 

The app should not imply official affiliation with Bandai, Sunrise, or the Gundam franchise unless such rights are obtained (not likely).