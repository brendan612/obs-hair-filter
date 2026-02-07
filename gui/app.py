import tkinter as tk
from tkinter import ttk, colorchooser
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional

from src.pipeline import Pipeline
from src.config import AppConfig
from src.color_transform import HAIR_COLOR_PRESETS


class HairColorApp(tk.Tk):
    """
    Main application window with live preview, color presets,
    custom color picker, and parameter controls.

    Threading model:
    - Main thread: tkinter event loop
    - Pipeline thread: daemon thread running pipeline.run_loop()
    - Frame handoff: pipeline stores latest frame; tkinter polls via after()
    """

    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    def __init__(self, config: AppConfig):
        super().__init__()
        self.title("OBS Hair Color Filter")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._config = config
        self._pipeline: Optional[Pipeline] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        # Track selected color name for highlighting
        self._selected_preset: Optional[str] = None

        self._build_ui()
        self._update_preview_loop()

    def _build_ui(self):
        # --- Preview ---
        preview_frame = ttk.LabelFrame(self, text="Preview", padding=5)
        preview_frame.pack(padx=10, pady=(10, 5))

        self._preview_canvas = tk.Canvas(
            preview_frame,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            bg="black",
            highlightthickness=0,
        )
        self._preview_canvas.pack()
        # Draw placeholder text
        self._preview_canvas.create_text(
            self.PREVIEW_WIDTH // 2,
            self.PREVIEW_HEIGHT // 2,
            text='Click "Start" to begin',
            fill="gray",
            font=("Segoe UI", 14),
            tags="placeholder",
        )

        # --- Controls row ---
        controls_frame = ttk.Frame(self, padding=5)
        controls_frame.pack(padx=10, pady=5, fill="x")

        # Start/Stop button
        self._start_btn = ttk.Button(
            controls_frame, text="Start", command=self._toggle_pipeline
        )
        self._start_btn.pack(side="left", padx=(0, 10))

        # Camera selector
        ttk.Label(controls_frame, text="Camera:").pack(side="left")
        self._camera_var = tk.IntVar(value=self._config.camera_index)
        camera_spin = ttk.Spinbox(
            controls_frame,
            from_=0,
            to=5,
            width=3,
            textvariable=self._camera_var,
        )
        camera_spin.pack(side="left", padx=(2, 10))

        # FPS display
        self._fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(controls_frame, textvariable=self._fps_var).pack(side="right")

        # --- Color presets ---
        color_frame = ttk.LabelFrame(self, text="Hair Color", padding=5)
        color_frame.pack(padx=10, pady=5, fill="x")

        preset_grid = ttk.Frame(color_frame)
        preset_grid.pack(fill="x")

        self._preset_buttons = {}
        for i, (name, lab) in enumerate(HAIR_COLOR_PRESETS.items()):
            row, col = divmod(i, 6)
            # Convert LAB to RGB for button color display
            rgb = self._lab_to_rgb(lab)
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            btn = tk.Button(
                preset_grid,
                text=name.replace(" ", "\n"),
                bg=hex_color,
                fg=self._contrast_text(rgb),
                width=12,
                height=2,
                font=("Segoe UI", 7),
                relief="raised",
                bd=2,
                command=lambda n=name, l=lab: self._select_preset(n, l),
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            self._preset_buttons[name] = btn

        for col_idx in range(6):
            preset_grid.columnconfigure(col_idx, weight=1)

        # Custom color button
        custom_row = ttk.Frame(color_frame)
        custom_row.pack(fill="x", pady=(5, 0))

        self._custom_btn = ttk.Button(
            custom_row, text="Custom Color...", command=self._pick_custom_color
        )
        self._custom_btn.pack(side="left")

        # Current color swatch
        self._swatch = tk.Label(custom_row, text="  ", width=4, relief="sunken")
        self._swatch.pack(side="left", padx=(10, 0))
        self._update_swatch(self._config.target_color_lab)

        # --- Sliders ---
        slider_frame = ttk.LabelFrame(self, text="Settings", padding=5)
        slider_frame.pack(padx=10, pady=5, fill="x")

        # Intensity slider
        ttk.Label(slider_frame, text="Intensity:").grid(row=0, column=0, sticky="w")
        self._intensity_var = tk.DoubleVar(value=self._config.color_intensity)
        intensity_scale = ttk.Scale(
            slider_frame,
            from_=0.0,
            to=1.0,
            variable=self._intensity_var,
            orient="horizontal",
            command=self._on_intensity_change,
        )
        intensity_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self._intensity_label = ttk.Label(
            slider_frame, text=f"{self._config.color_intensity:.0%}"
        )
        self._intensity_label.grid(row=0, column=2, sticky="e")

        # Temporal smoothing slider
        ttk.Label(slider_frame, text="Smoothing:").grid(row=1, column=0, sticky="w")
        self._smoothing_var = tk.DoubleVar(value=self._config.temporal_alpha)
        smoothing_scale = ttk.Scale(
            slider_frame,
            from_=0.0,
            to=1.0,
            variable=self._smoothing_var,
            orient="horizontal",
            command=self._on_smoothing_change,
        )
        smoothing_scale.grid(row=1, column=1, sticky="ew", padx=5)
        self._smoothing_label = ttk.Label(
            slider_frame, text=f"{self._config.temporal_alpha:.0%}"
        )
        self._smoothing_label.grid(row=1, column=2, sticky="e")

        # Edge softness slider (blur kernel)
        ttk.Label(slider_frame, text="Edge Softness:").grid(row=2, column=0, sticky="w")
        self._blur_var = tk.IntVar(value=self._config.mask_blur_kernel)
        blur_scale = ttk.Scale(
            slider_frame,
            from_=1,
            to=51,
            variable=self._blur_var,
            orient="horizontal",
            command=self._on_blur_change,
        )
        blur_scale.grid(row=2, column=1, sticky="ew", padx=5)
        self._blur_label = ttk.Label(
            slider_frame, text=str(self._config.mask_blur_kernel)
        )
        self._blur_label.grid(row=2, column=2, sticky="e")

        slider_frame.columnconfigure(1, weight=1)

    def _toggle_pipeline(self):
        if self._pipeline and self._pipeline.is_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self):
        self._config.camera_index = self._camera_var.get()
        self._pipeline = Pipeline(self._config)

        try:
            self._pipeline.start()
        except Exception as e:
            self._show_error(f"Failed to start: {e}")
            self._pipeline = None
            return

        self._pipeline.on_frame = self._on_pipeline_frame
        self._pipeline_thread = threading.Thread(
            target=self._pipeline.run_loop, daemon=True
        )
        self._pipeline_thread.start()
        self._start_btn.configure(text="Stop")

    def _stop_pipeline(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._start_btn.configure(text="Start")
        self._fps_var.set("FPS: --")
        # Restore placeholder
        self._preview_canvas.delete("preview")
        self._preview_canvas.create_text(
            self.PREVIEW_WIDTH // 2,
            self.PREVIEW_HEIGHT // 2,
            text='Click "Start" to begin',
            fill="gray",
            font=("Segoe UI", 14),
            tags="placeholder",
        )

    def _on_pipeline_frame(self, frame_bgr: np.ndarray):
        """Called from pipeline thread. Stores frame for GUI polling."""
        with self._frame_lock:
            self._latest_frame = frame_bgr

    def _update_preview_loop(self):
        """Polls latest frame and updates preview. Runs on main thread via after()."""
        frame = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None

        if frame is not None:
            # Remove placeholder text on first frame
            self._preview_canvas.delete("placeholder")

            # Resize for preview display
            preview = cv2.resize(
                frame, (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            # BGR -> RGB -> PIL -> PhotoImage
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            self._photo_image = ImageTk.PhotoImage(pil_image)
            self._preview_canvas.delete("preview")
            self._preview_canvas.create_image(
                0, 0, anchor="nw", image=self._photo_image, tags="preview"
            )

        # Update FPS display
        if self._pipeline and self._pipeline.is_running:
            self._fps_var.set(f"FPS: {self._pipeline.current_fps:.1f}")

        # Poll every 16ms (~60Hz)
        self.after(16, self._update_preview_loop)

    def _select_preset(self, name: str, lab: tuple):
        """Apply a color preset."""
        self._selected_preset = name
        self._config.target_color_lab = lab
        self._update_swatch(lab)

        # Highlight selected button
        for btn_name, btn in self._preset_buttons.items():
            btn.configure(
                relief="raised" if btn_name != name else "sunken",
                bd=2 if btn_name != name else 3,
            )

        if self._pipeline and self._pipeline.is_running:
            self._pipeline.update_color(*lab)

    def _pick_custom_color(self):
        """Open color chooser dialog."""
        result = colorchooser.askcolor(title="Choose Hair Color")
        if result and result[0]:
            r, g, b = (int(c) for c in result[0])
            # Convert RGB to LAB
            pixel = np.uint8([[[r, g, b]]])
            lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0]
            lab_tuple = (int(lab[0]), int(lab[1]), int(lab[2]))

            self._config.target_color_lab = lab_tuple
            self._selected_preset = None
            self._update_swatch(lab_tuple)

            # Unhighlight all presets
            for btn in self._preset_buttons.values():
                btn.configure(relief="raised", bd=2)

            if self._pipeline and self._pipeline.is_running:
                self._pipeline.update_color(*lab_tuple)

    def _update_swatch(self, lab: tuple):
        rgb = self._lab_to_rgb(lab)
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        self._swatch.configure(bg=hex_color)

    def _on_intensity_change(self, _event=None):
        val = self._intensity_var.get()
        self._intensity_label.configure(text=f"{val:.0%}")
        self._config.color_intensity = val
        if self._pipeline and self._pipeline.is_running:
            self._pipeline.update_color(
                *self._config.target_color_lab, intensity=val
            )

    def _on_smoothing_change(self, _event=None):
        val = self._smoothing_var.get()
        self._smoothing_label.configure(text=f"{val:.0%}")
        self._config.temporal_alpha = val
        if self._pipeline and self._pipeline.is_running:
            self._pipeline.update_segmenter(temporal_alpha=val)

    def _on_blur_change(self, _event=None):
        val = self._blur_var.get()
        # Ensure odd kernel
        if val % 2 == 0:
            val += 1
        self._blur_label.configure(text=str(val))
        self._config.mask_blur_kernel = val
        if self._pipeline and self._pipeline.is_running:
            self._pipeline.update_segmenter(blur_kernel=val)

    def _on_close(self):
        self._stop_pipeline()
        self._config.save()
        self.destroy()

    @staticmethod
    def _lab_to_rgb(lab: tuple) -> tuple:
        """Convert LAB tuple to RGB tuple for display."""
        pixel = np.uint8([[[lab[0], lab[1], lab[2]]]])
        rgb = cv2.cvtColor(pixel, cv2.COLOR_LAB2RGB)[0][0]
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    @staticmethod
    def _contrast_text(rgb: tuple) -> str:
        """Return black or white text for contrast against bg color."""
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return "white" if luminance < 128 else "black"

    @staticmethod
    def _show_error(message: str):
        from tkinter import messagebox
        messagebox.showerror("Error", message)
