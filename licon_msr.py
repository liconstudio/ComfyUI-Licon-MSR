import cv2
import numpy as np
import torch
from PIL import Image


class LiconMSR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "frame_count": ([17, 25, 33, 41], {"default": 17}),
            },
            "optional": {
                "1": ("IMAGE",),
                "2": ("IMAGE",),
                "3": ("IMAGE",),
                "4": ("IMAGE",),
                "background": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "create_video"
    CATEGORY = "Licon/MSR"

    def create_video(self, width, height, frame_count, background=None, **kwargs):
        if background is None:
            raise ValueError("background input is required")

        images = []
        for name in ("1", "2", "3", "4"):
            image = kwargs.get(name)
            if image is not None:
                images.append(self._prepare_image(image, (width, height)))

        images.append(self._prepare_image(background, (width, height)))
        frames = self._expand_frames(images, frame_count)
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return (output,)

    @staticmethod
    def _tensor_to_rgb_array(image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        return np.ascontiguousarray(image)

    @staticmethod
    def _prepare_image(image, target_size):
        image_array = LiconMSR._tensor_to_rgb_array(image)
        pil_image = Image.fromarray(image_array).convert("RGB")
        image_array = np.array(pil_image)
        if image_array.shape[1] == target_size[0] and image_array.shape[0] == target_size[1]:
            return np.ascontiguousarray(image_array)
        return cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _expand_frames(images, frame_count):
        base_count = frame_count // len(images)
        remainder = frame_count % len(images)
        frames = []
        for index, image in enumerate(images):
            repeats = base_count + (1 if index < remainder else 0)
            frames.extend([image] * repeats)
        return frames

NODE_CLASS_MAPPINGS = {
    "LiconMSR": LiconMSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LiconMSR": "Licon MSR",
}
