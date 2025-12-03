import tkinter as tk
from typing import Callable


class ControlPanel:
    def __init__(
        self,
        parent: tk.Widget,
        on_open: Callable[[], None],
        on_save: Callable[[], None],
        on_detect: Callable[[], None],
        on_add_roi: Callable[[], None],
        on_erase: Callable[[], None],
        on_mode_change: Callable[[str], None],
        on_selection_change: Callable[[], None],
    ):
        self.frame = tk.Frame(parent, padx=10, pady=10)
        self.frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Mode selection
        self.mode_var = tk.StringVar(value="class")
        tk.Label(self.frame, text="Mode").pack(anchor="w")
        modes = [("Class Code", "class"), ("Lama", "lama")]
        for label, value in modes:
            rb = tk.Radiobutton(
                self.frame,
                text=label,
                variable=self.mode_var,
                value=value,
                command=lambda v=value: on_mode_change(v),
            )
            rb.pack(anchor="w")

        tk.Label(self.frame, text="Controls", pady=6).pack(anchor="w")
        tk.Button(self.frame, text="Open Image", command=on_open, width=18).pack(pady=2)
        tk.Button(self.frame, text="Save Result", command=on_save, width=18).pack(pady=2)
        tk.Button(self.frame, text="Run Detection", command=on_detect, width=18).pack(pady=6)
        tk.Button(self.frame, text="Add Manual ROI", command=on_add_roi, width=18).pack(pady=2)
        tk.Button(self.frame, text="Erase Selected", command=on_erase, width=18).pack(pady=6)

        tk.Label(self.frame, text="Objects / ROIs", pady=6).pack(anchor="w")
        self.listbox = tk.Listbox(self.frame, selectmode=tk.MULTIPLE, width=24, height=12, exportselection=False)
        self.listbox.pack(fill=tk.X, pady=2)
        self.listbox.bind("<<ListboxSelect>>", lambda _event: on_selection_change())

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.frame, textvariable=self.status_var, fg="#228be6").pack(anchor="w", pady=8)

    def set_items(self, items: list[str], selected_indices: list[int]) -> None:
        self.listbox.delete(0, tk.END)
        for item in items:
            self.listbox.insert(tk.END, item)
        for idx in selected_indices:
            self.listbox.selection_set(idx)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def widget(self) -> tk.Frame:
        return self.frame
