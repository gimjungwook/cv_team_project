import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detect import load_default_detector, YOLODetector
from core.state import AppState
from ui.canvas import ImageCanvas
from ui.controls import ControlPanel


MAX_IMAGE_SIZE: Tuple[int, int] = (1920, 1080)
MAX_DISPLAY_SIZE: Tuple[int, int] = (1200, 900)
MAX_OBJECT_AREA_RATIO: float = 0.30  # per PRD limitation


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _resize_if_needed(image_bgr: np.ndarray, max_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Downscale the working image if it exceeds max_size."""
    h, w = image_bgr.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale
    return image_bgr, 1.0


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Object Eraser (CV Project)")
        self.state = AppState()
        self.detector: Optional[YOLODetector] = None
        self.roi_drawing = False
        self._drag_start: tuple[int, int] | None = None

        # Layout: left canvas, right controls
        left_frame = tk.Frame(self.root, padx=6, pady=6)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = ImageCanvas(left_frame)

        self.controls = ControlPanel(
            parent=self.root,
            on_open=self.open_image,
            on_save=self.save_image,
            on_detect=self.run_detection,
            on_add_roi=self.add_manual_roi,
            on_erase=self.erase_selected,
            on_mode_change=self.change_mode,
            on_selection_change=self.on_selection_change,
        )
        canvas_widget = self.canvas.widget()
        canvas_widget.bind("<ButtonPress-1>", self.on_canvas_press)
        canvas_widget.bind("<B1-Motion>", self.on_canvas_drag)
        canvas_widget.bind("<ButtonRelease-1>", self.on_canvas_release)

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            image_bgr = _read_image(Path(file_path))
            image_bgr, _ = _resize_if_needed(image_bgr, MAX_IMAGE_SIZE)
            self.state.set_image(image_bgr, Path(file_path), MAX_DISPLAY_SIZE)
            self.canvas.render_image(self.state)
            self.canvas.draw_overlays(self.state)
            self._update_items_view()
            self.set_status(f"Loaded: {Path(file_path).name}")
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", str(exc))
            self.set_status("Failed to load image")

    def save_image(self) -> None:
        if not self.state.has_image():
            messagebox.showinfo("Info", "Load an image first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("Bitmap", "*.bmp")],
        )
        if not save_path:
            return
        try:
            cv2.imwrite(save_path, self.state.image_bgr)
            self.set_status(f"Saved to {Path(save_path).name}")
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Save failed: {exc}")
            self.set_status("Save failed")

    def change_mode(self, mode: str) -> None:
        self.state.mode = mode
        self.set_status(f"Mode changed to {mode}")

    def run_detection(self) -> None:
        if not self.state.has_image():
            messagebox.showinfo("Info", "Load an image first.")
            return
        try:
            if self.detector is None:
                self.detector = load_default_detector(Path("models"))
            raw_detections = self.detector.detect(self.state.image_bgr)
            self.state.detections = self._filter_large_detections(raw_detections)
            manual_selection = {key for key in self.state.selected_items if key.startswith("roi:")}
            detection_selection = {f"det:{i}" for i in range(len(self.state.detections))}
            self.state.selected_items = manual_selection | detection_selection
            skipped = len(raw_detections) - len(self.state.detections)
            if skipped > 0:
                self.set_status(f"Detections: {len(self.state.detections)} (skipped {skipped} >30% area)")
            else:
                self.set_status(f"Detections: {len(self.state.detections)}")
            self.canvas.draw_overlays(self.state)
            self._update_items_view()
        except FileNotFoundError as exc:
            messagebox.showerror("Model missing", str(exc))
            self.set_status("Detector not ready")
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Detection failed: {exc}")
            self.set_status("Detection failed")

    def add_manual_roi(self) -> None:
        if not self.state.has_image():
            messagebox.showinfo("Info", "Load an image first.")
            return
        self.roi_drawing = True
        self.set_status("Drag on the image to add ROI")

    def erase_selected(self) -> None:
        if not self.state.has_image():
            messagebox.showinfo("Info", "Load an image first.")
            return
        selections = self.state.selected_items
        if not selections:
            messagebox.showinfo("Info", "Select detections or ROIs to erase.")
            return
        try:
            boxes = self._collect_selected_boxes()
            if not boxes:
                messagebox.showinfo("Info", "Nothing to erase. Run detection or add an ROI.")
                return
            if self.state.mode == "lama":
                messagebox.showinfo("Info", "Lama 모드는 아직 연결되지 않아 Class 파이프라인으로 진행합니다.")
            updated = self._erase_with_grabcut_and_blur(boxes)
            self._update_image(updated)
            self.set_status("Erased selected areas")
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Erase failed: {exc}")
            self.set_status("Erase failed")

    def set_status(self, text: str) -> None:
        self.state.status = text
        self.controls.set_status(text)

    def on_selection_change(self) -> None:
        items = self._item_catalog()
        selected_indices = list(self.controls.listbox.curselection())
        new_selection = set()
        for idx in selected_indices:
            key = items[idx][0]
            new_selection.add(key)
        self.state.selected_items = new_selection
        self.canvas.draw_overlays(self.state)

    def on_canvas_press(self, event) -> None:  # type: ignore[override]
        if not self.roi_drawing or not self.state.has_image():
            return
        self._drag_start = (event.x, event.y)

    def on_canvas_drag(self, event) -> None:  # type: ignore[override]
        if not self.roi_drawing or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self.canvas.show_active_roi(x0, y0, event.x, event.y)

    def on_canvas_release(self, event) -> None:  # type: ignore[override]
        if not self.roi_drawing or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self.roi_drawing = False
        self._drag_start = None
        self.canvas.clear_active_roi()

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        ix0, iy0 = self.state.display_to_image(x_min, y_min)
        ix1, iy1 = self.state.display_to_image(x_max, y_max)
        w = max(ix1 - ix0, 1)
        h = max(iy1 - iy0, 1)
        if w < 5 or h < 5:
            self.set_status("ROI too small, ignored")
            return
        if self._is_bbox_too_big((ix0, iy0, w, h)):
            messagebox.showinfo("Info", "선택 영역이 이미지의 30%를 초과하여 무시되었습니다.")
            self.set_status("ROI too large (>30%), ignored")
            return
        self.state.manual_rois.append((ix0, iy0, w, h))
        key = f"roi:{len(self.state.manual_rois) - 1}"
        self.state.selected_items.add(key)
        self.canvas.draw_overlays(self.state)
        self._update_items_view()
        self.set_status(f"Added ROI #{len(self.state.manual_rois)}")

    def _item_catalog(self) -> List[tuple[str, str]]:
        """Return list of (key, label) pairs in the order shown in the UI."""
        items: List[tuple[str, str]] = []
        for idx, det in enumerate(self.state.detections):
            label = f"{det.label} {det.confidence * 100:.0f}%"
            items.append((f"det:{idx}", label))
        for idx, _roi in enumerate(self.state.manual_rois):
            items.append((f"roi:{idx}", f"Manual ROI #{idx + 1}"))
        return items

    def _update_items_view(self) -> None:
        catalog = self._item_catalog()
        labels = [label for _key, label in catalog]
        selected_indices = [
            idx for idx, (key, _label) in enumerate(catalog) if key in self.state.selected_items
        ]
        self.controls.set_items(labels, selected_indices)

    def _collect_selected_boxes(self) -> List[tuple[int, int, int, int]]:
        boxes: List[tuple[int, int, int, int]] = []
        for key in self.state.selected_items:
            if key.startswith("det:"):
                idx = int(key.split(":")[1])
                if 0 <= idx < len(self.state.detections):
                    boxes.append(self.state.detections[idx].bbox)
            elif key.startswith("roi:"):
                idx = int(key.split(":")[1])
                if 0 <= idx < len(self.state.manual_rois):
                    boxes.append(self.state.manual_rois[idx])
        return boxes

    def _erase_with_grabcut_and_blur(self, boxes: List[tuple[int, int, int, int]]) -> np.ndarray:
        """Class-mode erase: GrabCut on selected boxes, then blend with blurred background."""
        assert self.state.image_bgr is not None
        image = self.state.image_bgr
        h, w = image.shape[:2]
        accumulated_mask = np.zeros((h, w), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        for x, y, bw, bh in boxes:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w - 1), min(y + bh, h - 1)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, (x1, y1, x2 - x1, y2 - y1), bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, fg)

        if not np.any(accumulated_mask):
            raise ValueError("GrabCut failed to produce a mask; adjust selection or ROI.")

        dilated_mask = cv2.dilate(accumulated_mask, kernel, iterations=1)

        if self.state.mode == "lama":
            # TODO: Lama 모델 연결 시 이 분기에서 마스크를 사용해 인페인팅을 대체.
            pass

        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        result = image.copy()
        result[dilated_mask > 0] = blurred[dilated_mask > 0]
        return result

    def _filter_large_detections(self, detections: List["DetectionResult"]) -> List["DetectionResult"]:
        max_area = self._max_allowed_area()
        filtered: List["DetectionResult"] = []
        for det in detections:
            x, y, w, h = det.bbox
            if w * h <= max_area:
                filtered.append(det)
        return filtered

    def _max_allowed_area(self) -> float:
        img_w, img_h = self.state.image_size()
        return img_w * img_h * MAX_OBJECT_AREA_RATIO

    def _is_bbox_too_big(self, bbox: tuple[int, int, int, int]) -> bool:
        _x, _y, w, h = bbox
        return w * h > self._max_allowed_area()

    def _update_image(self, new_bgr: np.ndarray) -> None:
        """Replace image and refresh display/overlays."""
        self.state.set_image(new_bgr, self.state.image_path, MAX_DISPLAY_SIZE)
        self.canvas.render_image(self.state)
        self.canvas.draw_overlays(self.state)
        self._update_items_view()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
