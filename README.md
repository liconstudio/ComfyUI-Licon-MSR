# ComfyUI-Licon-MSR

A ComfyUI custom node for creating fixed-frame MP4 reference videos from multiple subject images and a background image, designed for LTX 2.3 MSR (Multiple-Subject-Reference) LoRA workflows.

## Latest Update

- Updated frame allocation to align reference images with LTX's 8x temporal VAE compression boundaries.
- Subject images (`1`-`4`) are resized proportionally and always kept complete; unused canvas space is filled with white.
- The background image is resized proportionally to cover the full canvas and center-cropped when necessary.
- Added `49`, `57`, and `65` frame options.

## Node

- Display name: `Licon MSR`
- Category: `Licon/MSR`

## Inputs

- `1`: optional image
- `2`: optional image
- `3`: optional image
- `4`: optional image
- `background`: required image
- `width`: output video width
- `height`: output video height
- `frame_count`: `17`, `25`, `33`, `41`, `49`, `57`, or `65`

## Behavior

Images are processed in this fixed order:

```text
1 -> 2 -> 3 -> 4 -> background
```

Disconnected `1` to `4` inputs are skipped. `background` is always required and always placed last.

Subject images are proportionally fitted inside the configured dimensions without cropping and centered on a white canvas. The background proportionally covers the full canvas and is center-cropped if its aspect ratio differs. Frames are assigned in LTX-aligned temporal groups, and the node outputs an `IMAGE` frame sequence for downstream video processing.

## Installation

Clone this repository into the ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liconstudio/ComfyUI-Licon-MSR
```

Install requirements if needed:

```bash
pip install -r ComfyUI-Licon-MSR/requirements.txt
```

Restart ComfyUI after installation.
