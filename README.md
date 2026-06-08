# Gunpla Studio

Gunpla Studio is a computer vision web app for:
single-class Gunpla detection via instance segmentation
then background replacement.
It is designed to run locally or on Hugging Face Spaces with Gradio.

The project includes three segmentation implementations behind one interface as per requirements:

- `NaiveBaselineSegmenter`: simple center-biased foreground estimate.
- `ClassicalMLSegmenter`: OpenCV GrabCut segmentation.
- `YOLOSegSegmenter`: YOLO11 segmentation inference for fine-tuned Gunpla masks.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── setup.py - run this first
├── main.py - start gradio app
├── report.md - required report for submission
├── scripts
│   ├── make_dataset.py
│   ├── build_features.py
│   └── train_model.py
├── gunpla_studio
│   ├── __init__.py
│   ├── backgrounds.py
│   ├── image_utils.py
│   └── segmenters.py
├── models - generated models go here
├── data
│   ├── raw
│   ├── processed
│   └── outputs
└── notebooks
```

## Setup

Use Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cu128.txt
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

`requirements-gpu-cu128.txt` installs PyTorch from the CUDA 12.8 wheel index for NVIDIA GPU training. Install it before `requirements.txt`; otherwise `ultralytics` may pull the CPU PyTorch wheel from PyPI. After installation, verify GPU visibility with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If `torch.cuda.is_available()` is `False`, confirm that your NVIDIA driver supports CUDA 12.8 wheels or install the PyTorch build recommended for your driver/GPU from the official PyTorch selector.

## Run The App

```bash
python main.py
```

The app will start a local Gradio server. Upload a Gunpla image or take one with a phone camera, choose a segmentation method, then choose a default or uploaded background.

To add reusable backgrounds, place image files in the top-level `backgrounds` folder. The app also loads packaged backgrounds from `gunpla_studio/backgrounds`. Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. The app displays each background by filename without the extension.

## Data

Expected training data format is YOLO segmentation format:

```text
data/processed/gunpla-yolov7/
├── data.yaml
├── train
│   ├── images
│   └── labels
├── valid
│   ├── images
│   └── labels
└── test
    ├── images
    └── labels
```

Each label file should contain one class, `gunpla`, with polygon coordinates normalized to image dimensions.

If you have a Roboflow/Label Studio/CVAT export ZIP, place it in `data/raw` and run:

```bash
python scripts/make_dataset.py --zip-path data/raw/your_export.zip --output-dir data/processed/gunpla-yolov7
```

## Train YOLO11n

Check the dataset first:

```bash
python scripts/train_model.py --validate-only
```

Start transfer-learning training on GPU 0:

```bash
python scripts/train_model.py --device 0
```

Training uses light augmentation by default: modest color jitter, small geometric transforms, horizontal flips, very low mosaic, and light random erasing. To run an ablation without augmentation:

```bash
python scripts/train_model.py --device 0 --augment-mode none
```

Available modes are `none`, `light`, and `strong`. `strong` uses heavier mosaic, mixup, copy-paste, and geometric transforms; it has not performed well so far on this small segmentation dataset.

To force a specific GPU:

```bash
python scripts/train_model.py --data data/processed/gunpla-yolov7/data.yaml --epochs 50 --imgsz 640 --batch 4 --workers 0 --device 0
```

The default dataset is `data/processed/gunpla-yolov7/data.yaml`. The script inspects the labels before training. This dataset has YOLO polygon segmentation labels, so the default transfer-learning base model is `yolo11n-seg.pt`. If Roboflow split paths need correction, the script writes an Ultralytics-ready data file under `data/processed/gunpla-yolov7-ultralytics`.

On Windows, keep `--workers 0` unless you have plenty of RAM and page file space. Multiple dataloader worker processes can each load PyTorch CUDA DLLs and trigger `[WinError 1455] The paging file is too small`.

To make a cleaned copy that removes overlapping duplicate train/validation masks and keeps the largest overlapping mask:

```bash
python scripts/clean_overlapping_masks.py --overwrite
```

Train from the cleaned copy with:

```bash
python scripts/train_model.py --data data/processed/gunpla-yolov7-largest-mask/data.yaml --device 0 --workers 0
```

The best model is copied to:

```text
models/<run-name>.pt
```

By default, each training run gets a unique name like `gunpla_yolo11s_seg_img768_b2_aug_20260607_224500`, so previous results are not overwritten. To set a shorter name manually:

```bash
python scripts/train_model.py --device 0 --model yolo11s-seg.pt --run-name gunpla_yolo11s_light_aug
```

Set `GUNPLA_MODEL_PATH` to use a different trained model in the app:

```bash
$env:GUNPLA_MODEL_PATH="models/gunpla_yolo11n_seg.pt"
python main.py
```

## Notes

- The app works without a trained YOLO model by using the naive or classical methods.
- Deep learning inference requires a trained `models/gunpla_yolo11n_seg.pt` file.
- This is a single-class project: the only target class is `gunpla`.
