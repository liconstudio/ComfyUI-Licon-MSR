import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


class LiconMSR:
    FRAME_COUNTS = (17, 25, 33, 41, 49, 57, 65)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "frame_count": ([str(value) for value in cls.FRAME_COUNTS], {"default": "41"}),
            },
            "optional": {
                "1": ("IMAGE",),
                "2": ("IMAGE",),
                "3": ("IMAGE",),
                "4": ("IMAGE",),
                "background": ("IMAGE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, frame_count):
        try:
            frame_count = int(frame_count)
        except (TypeError, ValueError):
            return "frame_count must be one of: " + ", ".join(map(str, cls.FRAME_COUNTS))

        if frame_count not in cls.FRAME_COUNTS:
            return "frame_count must be one of: " + ", ".join(map(str, cls.FRAME_COUNTS))

        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "create_video"
    CATEGORY = "Licon/MSR"

    def create_video(self, width, height, frame_count, background=None, **kwargs):
        frame_count = int(frame_count)

        if background is None:
            raise ValueError("background input is required")

        subjects = []
        for slot in ("1", "2", "3", "4"):
            image = kwargs.get(slot)
            if image is not None:
                subjects.append(self._prepare_image(image, (width, height), preserve_full=True))

        background_image = self._prepare_image(background, (width, height), preserve_full=False)
        frames = self._expand_frames(subjects, background_image, frame_count)
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
    def _prepare_image(image, target_size, preserve_full=False):
        image_array = LiconMSR._tensor_to_rgb_array(image)
        source_height, source_width = image_array.shape[:2]
        target_width, target_height = target_size

        if source_width == target_width and source_height == target_height:
            return np.ascontiguousarray(image_array)

        if preserve_full:
            # Subject images are always fitted proportionally and centered on
            # white. They are never cropped, regardless of their source size.
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = max(1, min(target_width, round(source_width * scale)))
            resized_height = max(1, min(target_height, round(source_height * scale)))
            resized = LiconMSR._resize(image_array, resized_width, resized_height)
            canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
            left = (target_width - resized_width) // 2
            top = (target_height - resized_height) // 2
            canvas[top:top + resized_height, left:left + resized_width] = resized
            return np.ascontiguousarray(canvas)

        # Preserve aspect ratio, resize until the target is fully covered, then
        # remove the excess equally from both sides around the image center.
        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, round(source_width * scale))
        resized_height = max(target_height, round(source_height * scale))
        resized = LiconMSR._resize(image_array, resized_width, resized_height)
        left = (resized_width - target_width) // 2
        top = (resized_height - target_height) // 2
        return np.ascontiguousarray(
            resized[top:top + target_height, left:left + target_width]
        )

    @staticmethod
    def _resize(image_array, width, height):
        if cv2 is not None:
            interpolation = (
                cv2.INTER_AREA
                if width < image_array.shape[1] or height < image_array.shape[0]
                else cv2.INTER_LANCZOS4
            )
            return cv2.resize(image_array, (width, height), interpolation=interpolation)

        chw = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(
            chw,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return np.ascontiguousarray(
            resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        )

    @staticmethod
    def _expand_frames(subjects, background, frame_count):
        latent_count = _estimate_ref_latent_frames(frame_count)
        frames = [background] * frame_count

        if not subjects:
            return frames

        subject_budget = max(0, latent_count - 1)
        if subject_budget >= len(subjects):
            counts = _allocate_subject_latent_counts(len(subjects), subject_budget)
            cursor = 0
            for image, count in zip(subjects, counts):
                start, end = _latent_to_frame_range(cursor, cursor + count - 1)
                cursor += count
                for frame_index in range(max(0, start), min(frame_count - 1, end) + 1):
                    frames[frame_index] = image
        else:
            subject_frame_count = max(
                1,
                _latent_to_frame_range(0, max(0, latent_count - 2))[1] + 1,
            )
            for index, image in enumerate(subjects):
                start = int(index * subject_frame_count / len(subjects))
                end = int((index + 1) * subject_frame_count / len(subjects)) - 1
                if index == len(subjects) - 1:
                    end = subject_frame_count - 1
                for frame_index in range(start, min(frame_count - 1, max(start, end)) + 1):
                    frames[frame_index] = image

        return frames


def _estimate_ref_latent_frames(source_frame_count):
    if source_frame_count <= 1:
        return max(1, source_frame_count)
    return int(round((source_frame_count - 1) / 8.0)) + 1


def _latent_to_frame_range(latent_start, latent_end):
    latent_start = int(latent_start)
    latent_end = int(latent_end)
    frame_start = 0 if latent_start <= 0 else 1 + (latent_start - 1) * 8
    frame_end = 0 if latent_end <= 0 else latent_end * 8
    return frame_start, frame_end


def _allocate_subject_latent_counts(num_subjects, subject_budget):
    counts = [1] * num_subjects
    extra = max(0, int(subject_budget) - num_subjects)

    if extra > 0:
        counts[0] += 1
        extra -= 1

    index = 1
    while extra > 0 and num_subjects > 1 and any(count < 2 for count in counts[1:]):
        if counts[index] < 2:
            counts[index] += 1
            extra -= 1
        index = index + 1 if index + 1 < num_subjects else 1

    if extra > 0 and counts[0] < 3:
        counts[0] += 1
        extra -= 1

    index = 0
    while extra > 0:
        counts[index] += 1
        extra -= 1
        index = (index + 1) % num_subjects

    return counts


NODE_CLASS_MAPPINGS = {
    "LiconMSR": LiconMSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LiconMSR": "Licon MSR",
}
