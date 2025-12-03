from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image


def _resize_to_fit(image_rgb: np.ndarray, max_size: Tuple[int, int]) -> Tuple[Image.Image, float]:
    """Resize an RGB image to fit inside max_size while keeping aspect ratio."""
    height, width = image_rgb.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / width, max_h / height, 1.0)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        pil_image = Image.fromarray(image_rgb).resize(new_size, Image.LANCZOS)
    else:
        pil_image = Image.fromarray(image_rgb)
    return pil_image, scale


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h in image coords


@dataclass
class AppState:
    image_path: Optional[Path] = None
    image_bgr: Optional[np.ndarray] = None
    display_image: Optional[Image.Image] = None
    scale: float = 1.0
    mode: str = "class"  # "class" or "lama"
    detections: List[DetectionResult] = field(default_factory=list)
    selected_items: Set[str] = field(default_factory=set)  # keys like "det:0", "roi:1"
    manual_rois: List[Tuple[int, int, int, int]] = field(default_factory=list)
    status: str = "Ready"

    def has_image(self) -> bool:
        return self.image_bgr is not None

    def set_image(self, image_bgr: np.ndarray, path: Optional[Path], max_size: Tuple[int, int]) -> None:
        """Load image into state and prepare display version."""
        self.image_path = path
        self.image_bgr = image_bgr
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.display_image, self.scale = _resize_to_fit(image_rgb, max_size)
        self.reset_annotations()
        self.status = "Image loaded"

    def reset_annotations(self) -> None:
        self.detections = []
        self.selected_items = set()
        self.manual_rois = []

    def image_size(self) -> Tuple[int, int]:
        if self.image_bgr is None:
            return 0, 0
        h, w = self.image_bgr.shape[:2]
        return w, h

    def display_size(self) -> Tuple[int, int]:
        if self.display_image is None:
            return 0, 0
        return self.display_image.size

    def image_to_display(self, x: int, y: int) -> Tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        if self.scale == 0:
            return x, y
        return int(x / self.scale), int(y / self.scale)
