from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ncai_pipeline.config import BlankPageDetectionSettings


@dataclass(frozen=True)
class BlankPageDetectionResult:
    is_blank: bool
    reason: str
    metrics: dict[str, Any]

    def model_payload(self) -> dict[str, Any]:
        return {
            "is_blank": self.is_blank,
            "reason": self.reason,
            "metrics": self.metrics,
        }


def detect_blank_page(path: Path, settings: BlankPageDetectionSettings) -> BlankPageDetectionResult:
    if not settings.enabled:
        return BlankPageDetectionResult(False, "disabled", {"enabled": False})

    try:
        with Image.open(path) as image:
            grayscale = ImageOps.exif_transpose(image).convert("L")
            original_width, original_height = grayscale.size
            original_pixels = original_width * original_height
            if original_pixels < settings.min_pixels:
                return BlankPageDetectionResult(
                    False,
                    "below_min_pixels",
                    {
                        "enabled": True,
                        "width": original_width,
                        "height": original_height,
                        "pixel_count": original_pixels,
                        "min_pixels": settings.min_pixels,
                    },
                )
            grayscale.thumbnail((settings.sample_max_dimension, settings.sample_max_dimension))
            histogram = grayscale.histogram()
    except OSError as exc:
        return BlankPageDetectionResult(False, "image_read_error", {"enabled": True, "error": str(exc)})

    sampled_pixels = sum(histogram)
    nonwhite_pixels = sum(histogram[: settings.white_threshold])
    dark_pixels = sum(histogram[: settings.dark_threshold])
    luma_total = sum(value * count for value, count in enumerate(histogram))
    mean_luma = (luma_total / sampled_pixels) if sampled_pixels else 0.0
    nonwhite_ratio = (nonwhite_pixels / sampled_pixels) if sampled_pixels else 1.0
    dark_ratio = (dark_pixels / sampled_pixels) if sampled_pixels else 1.0
    is_blank = (
        nonwhite_ratio <= settings.max_nonwhite_ratio
        and dark_ratio <= settings.max_dark_ratio
        and mean_luma >= settings.min_mean_luma
    )
    metrics = {
        "enabled": True,
        "sampled_pixels": sampled_pixels,
        "nonwhite_pixels": nonwhite_pixels,
        "dark_pixels": dark_pixels,
        "nonwhite_ratio": round(nonwhite_ratio, 6),
        "dark_ratio": round(dark_ratio, 6),
        "mean_luma": round(mean_luma, 3),
        "white_threshold": settings.white_threshold,
        "dark_threshold": settings.dark_threshold,
        "max_nonwhite_ratio": settings.max_nonwhite_ratio,
        "max_dark_ratio": settings.max_dark_ratio,
        "min_mean_luma": settings.min_mean_luma,
    }
    reason = "blank_by_thresholds" if is_blank else "content_or_uncertain"
    return BlankPageDetectionResult(is_blank, reason, metrics)
