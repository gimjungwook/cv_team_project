from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from core.state import DetectionResult


def _load_names(names_path: Path) -> List[str]:
    with names_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class YOLODetector:
    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        names_path: Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError("YOLO config/weights not found. Place files under models/ and try again.")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.names = _load_names(names_path) if names_path.exists() else []
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: List[tuple[int, int, int, int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.conf_threshold:
                    continue
                center_x, center_y, bw, bh = (
                    int(detection[0] * w),
                    int(detection[1] * h),
                    int(detection[2] * w),
                    int(detection[3] * h),
                )
                x = max(int(center_x - bw / 2), 0)
                y = max(int(center_y - bh / 2), 0)
                boxes.append((x, y, bw, bh))
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        results: List[DetectionResult] = []
        for i in indices.flatten() if len(indices) > 0 else []:
            label = self.names[class_ids[i]] if class_ids[i] < len(self.names) else f"id{class_ids[i]}"
            results.append(DetectionResult(label=label, confidence=confidences[i], bbox=boxes[i]))
        return results


def load_default_detector(model_dir: Path) -> YOLODetector:
    """
    Convenience loader for YOLO Darknet files placed under models/.

    It will pick the first .cfg/.weights pair it can find, preferring matching
    stems (e.g., foo.cfg + foo.weights). A .names file is optional but recommended.
    """
    cfg_candidates = sorted(model_dir.rglob("*.cfg"))
    weight_candidates = sorted(model_dir.rglob("*.weights"))
    name_candidates = sorted(model_dir.rglob("*.names"))

    if not cfg_candidates or not weight_candidates:
        raise FileNotFoundError(
            "YOLO config/weights not found. Place your .cfg and .weights files under models/ "
            "and try again."
        )

    # Prefer matching stem pairs, otherwise fall back to the first available.
    cfg_path = cfg_candidates[0]
    weights_path = weight_candidates[0]
    for cfg in cfg_candidates:
        stem = cfg.stem
        match = cfg.with_suffix(".weights")
        if match.exists():
            cfg_path = cfg
            weights_path = match
            break

    names_path = cfg_path.with_suffix(".names")
    if not names_path.exists():
        # Fallback to coco.names or any .names file.
        names_path = model_dir / "coco.names"
        if not names_path.exists() and name_candidates:
            names_path = name_candidates[0]

    return YOLODetector(cfg_path=cfg_path, weights_path=weights_path, names_path=names_path)
