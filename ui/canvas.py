import tkinter as tk
from typing import Optional

from PIL import ImageTk

from core.state import AppState


class ImageCanvas:
    def __init__(self, parent: tk.Widget):
        self.canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._image_id: Optional[int] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._active_roi_id: Optional[int] = None

    def render_image(self, state: AppState) -> None:
        """Draw the current image on the canvas."""
        self.canvas.delete("all")
        if not state.display_image:
            return
        self._photo = ImageTk.PhotoImage(state.display_image)
        self._image_id = self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        width, height = state.display_size()
        self.canvas.config(scrollregion=(0, 0, width, height), width=width, height=height)

    def draw_overlays(self, state: AppState) -> None:
        """Draw detection boxes and manual ROIs."""
        self.canvas.delete("overlay")
        if not state.display_image:
            return
        for idx, det in enumerate(state.detections):
            x, y, w, h = det.bbox
            x1, y1 = state.image_to_display(x, y)
            x2, y2 = state.image_to_display(x + w, y + h)
            key = f"det:{idx}"
            color = "#3bd671" if key in state.selected_items else "#2e8bc0"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="overlay")
            self.canvas.create_text(x1 + 4, y1 + 12, text=det.label, anchor="w", fill=color, font=("Arial", 10), tags="overlay")
        for idx, roi in enumerate(state.manual_rois):
            x, y, w, h = roi
            x1, y1 = state.image_to_display(x, y)
            x2, y2 = state.image_to_display(x + w, y + h)
            key = f"roi:{idx}"
            color = "#ffb347" if key in state.selected_items else "#ff8c00"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=(4, 2), tags="overlay")
            self.canvas.create_text(x1 + 4, y1 + 12, text=f"ROI #{idx+1}", anchor="w", fill=color, font=("Arial", 10), tags="overlay")

    def show_active_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self._active_roi_id:
            self.canvas.delete(self._active_roi_id)
        self._active_roi_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#f7768e", width=2, dash=(4, 2), tags="overlay"
        )

    def clear_active_roi(self) -> None:
        if self._active_roi_id:
            self.canvas.delete(self._active_roi_id)
            self._active_roi_id = None

    def widget(self) -> tk.Canvas:
        return self.canvas
