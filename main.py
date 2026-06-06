from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from gunpla_studio.backgrounds import DEFAULT_BACKGROUNDS, make_default_background
from gunpla_studio.image_utils import composite_foreground, ensure_rgb_image
from gunpla_studio.segmenters import (
    ClassicalMLSegmenter,
    NaiveBaselineSegmenter,
    SegmentationResult,
    YOLOSegSegmenter,
)


MODEL_PATH = Path(os.getenv("GUNPLA_MODEL_PATH", "models/gunpla_yolo11n_seg.pt"))

SEGMENTERS = {
    "Naive baseline": NaiveBaselineSegmenter(),
    "Classical ML (GrabCut)": ClassicalMLSegmenter(),
    "Deep learning (YOLO11n-seg)": YOLOSegSegmenter(MODEL_PATH),
}


def run_studio(
    gunpla_image: Image.Image | np.ndarray | None,
    method: str,
    default_background: str,
    custom_background: Image.Image | np.ndarray | None,
) -> tuple[Image.Image | None, Image.Image | None, str]:
    if gunpla_image is None:
        return None, None, "Upload or capture a Gunpla image first."

    image = ensure_rgb_image(gunpla_image)
    segmenter = SEGMENTERS[method]
    result: SegmentationResult = segmenter.segment(image)

    background_source = custom_background
    if background_source is None:
        background = make_default_background(default_background, image.size)
    else:
        background = ensure_rgb_image(background_source).resize(image.size, Image.Resampling.LANCZOS)

    composite = composite_foreground(image, result.mask, background)
    mask_preview = Image.fromarray((result.mask * 255).astype(np.uint8), mode="L")
    status = f"{result.method}: {result.message}"
    return composite, mask_preview, status


with gr.Blocks(title="Gunpla Studio") as demo:
    gr.Markdown("# Gunpla Studio")
    gr.Markdown("Single-class Gunpla isolation and background replacement.")

    with gr.Row():
        with gr.Column(scale=1):
            gunpla_input = gr.Image(
                label="Gunpla image",
                sources=["upload", "webcam"],
                type="pil",
            )
            method = gr.Radio(
                choices=list(SEGMENTERS.keys()),
                value="Classical ML (GrabCut)",
                label="Segmentation method",
            )
            default_bg = gr.Dropdown(
                choices=list(DEFAULT_BACKGROUNDS.keys()),
                value="Studio gray",
                label="Default background",
            )
            custom_bg = gr.Image(
                label="Custom background",
                sources=["upload", "webcam"],
                type="pil",
            )
            run_button = gr.Button("Create composite", variant="primary")

        with gr.Column(scale=1):
            composite_output = gr.Image(label="Composite", type="pil")
            mask_output = gr.Image(label="Mask preview", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

    run_button.click(
        fn=run_studio,
        inputs=[gunpla_input, method, default_bg, custom_bg],
        outputs=[composite_output, mask_output, status],
    )


if __name__ == "__main__":
    demo.launch()
