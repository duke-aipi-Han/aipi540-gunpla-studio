from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from gunpla_studio.backgrounds import get_background_choices, load_background
from gunpla_studio.image_utils import (
    composite_overlay,
    ensure_rgb_image,
    make_foreground_overlay,
    resize_image_to_max_side,
)
from gunpla_studio.segmenters import (
    ClassicalMLSegmenter,
    NaiveBaselineSegmenter,
    SegmentationResult,
    YOLOSegSegmenter,
)


MODEL_PATH = Path(os.getenv("GUNPLA_MODEL_PATH", "models/gunpla_yolo11n_seg.pt"))
MAX_IMAGE_SIDE = int(os.getenv("GUNPLA_MAX_IMAGE_SIDE", "1280"))

SEGMENTERS = {
    "Naive baseline": NaiveBaselineSegmenter(),
    "Classical ML (GrabCut)": ClassicalMLSegmenter(),
    "Deep learning (YOLO11n-seg)": YOLOSegSegmenter(MODEL_PATH, imgsz=MAX_IMAGE_SIDE),
}
BACKGROUND_CHOICES = get_background_choices()
DEFAULT_BACKGROUND = BACKGROUND_CHOICES[0] if BACKGROUND_CHOICES else "Studio gray"


def run_studio(
    gunpla_image: Image.Image | np.ndarray | None,
    method: str,
    default_background: str,
    custom_background: Image.Image | np.ndarray | None,
) -> tuple[
    Image.Image | None,
    Image.Image | None,
    str,
    Image.Image | None,
    Image.Image | None,
    dict,
    dict,
]:
    if gunpla_image is None:
        return (
            None,
            None,
            "Upload or capture a Gunpla image first.",
            None,
            None,
            gr.update(open=True),
            gr.update(visible=False),
        )

    image = resize_image_to_max_side(ensure_rgb_image(gunpla_image), MAX_IMAGE_SIDE)
    segmenter = SEGMENTERS[method]
    result: SegmentationResult = segmenter.segment(image)

    if custom_background is None:
        background = load_background(default_background, image.size)
    else:
        background = resize_image_to_max_side(ensure_rgb_image(custom_background), MAX_IMAGE_SIDE)

    background = resize_image_to_max_side(background, MAX_IMAGE_SIDE)

    overlay = make_foreground_overlay(image, result.mask)
    composite = composite_overlay(background, overlay)
    mask_preview = Image.fromarray((result.mask * 255).astype(np.uint8), mode="L")
    status = f"{result.method}: {result.message}"
    return (
        composite,
        mask_preview,
        status,
        background,
        overlay,
        gr.update(open=False),
        gr.update(visible=True),
    )


def rotate_source(
    gunpla_image: Image.Image | np.ndarray | None,
    angle: int,
) -> Image.Image | None:
    if gunpla_image is None:
        return None
    return ensure_rgb_image(gunpla_image).rotate(angle, expand=True)


def update_overlay_transform(
    background: Image.Image | np.ndarray | None,
    overlay: Image.Image | np.ndarray | None,
    rotation_degrees: float,
    zoom: float,
    pan_x_percent: float,
    pan_y_percent: float,
) -> Image.Image | None:
    if background is None or overlay is None:
        return None
    return composite_overlay(
        ensure_rgb_image(background),
        overlay if isinstance(overlay, Image.Image) else Image.fromarray(overlay),
        rotation_degrees,
        zoom,
        pan_x_percent,
        pan_y_percent,
    )


with gr.Blocks(title="Gunpla Studio") as demo:
    gr.Markdown("# Gunpla Studio")
    gr.Markdown("Single-class Gunpla isolation and background replacement.")

    background_state = gr.State()
    overlay_state = gr.State()

    with gr.Accordion("1. Upload or capture Gunpla", open=True) as upload_section:
        gunpla_input = gr.Image(
            label="Gunpla image",
            sources=["upload", "webcam"],
            type="pil",
        )
        with gr.Row():
            rotate_left = gr.Button("Rotate left")
            rotate_right = gr.Button("Rotate right")

    default_bg = gr.Dropdown(
        choices=BACKGROUND_CHOICES,
        value=DEFAULT_BACKGROUND,
        label="2. Select Background",
    )
    run_button = gr.Button("3. Create composite", variant="primary")

    composite_output = gr.Image(label="Composite", type="pil")

    with gr.Group(visible=False) as transform_panel:
        overlay_rotation = gr.Slider(-180, 180, value=0, step=1, label="Gunpla rotation")
        overlay_zoom = gr.Slider(0.25, 2.5, value=1.0, step=0.05, label="Gunpla zoom")
        pan_x = gr.Slider(-50, 50, value=0, step=1, label="Gunpla pan left / right")
        pan_y = gr.Slider(-50, 50, value=0, step=1, label="Gunpla pan up / down")

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Accordion("Power user options", open=False):
        method = gr.Radio(
            choices=list(SEGMENTERS.keys()),
            value="Deep learning (YOLO11n-seg)",
            label="Segmentation method",
        )
        custom_bg = gr.Image(
            label="Custom background",
            sources=["upload", "webcam"],
            type="pil",
        )
        mask_output = gr.Image(label="Mask preview", type="pil")

    rotate_left.click(
        fn=lambda image: rotate_source(image, 90),
        inputs=gunpla_input,
        outputs=gunpla_input,
    )
    rotate_right.click(
        fn=lambda image: rotate_source(image, -90),
        inputs=gunpla_input,
        outputs=gunpla_input,
    )

    run_button.click(
        fn=run_studio,
        inputs=[gunpla_input, method, default_bg, custom_bg],
        outputs=[
            composite_output,
            mask_output,
            status,
            background_state,
            overlay_state,
            upload_section,
            transform_panel,
        ],
    )

    transform_inputs = [background_state, overlay_state, overlay_rotation, overlay_zoom, pan_x, pan_y]
    for control in [overlay_rotation, overlay_zoom, pan_x, pan_y]:
        control.change(
            fn=update_overlay_transform,
            inputs=transform_inputs,
            outputs=composite_output,
        )


if __name__ == "__main__":
    demo.launch()
