# ComfyUI-Licon-MSR

A ComfyUI custom node for creating fixed-frame MP4 reference videos from multiple subject images and a background image, designed for LTX 2.3 MSR (Multiple-Subject-Reference) LoRA workflows.

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
- `frame_count`: `17`, `25`, `33`, or `41`

## Behavior

Images are processed in this fixed order:

```text
1 -> 2 -> 3 -> 4 -> background
```

Disconnected `1` to `4` inputs are skipped. `background` is always required and always placed last.

Each input image is resized to the configured `width` and `height` before the video frames are created. The selected frame count is distributed across the connected images in order. The output is an MP4 file encoded at 24 fps.

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
