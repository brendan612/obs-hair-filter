import sys
import os


def get_base_path():
    """Get the base path for bundled resources (PyInstaller or dev)."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


def get_app_dir():
    """Get the directory where the .exe lives (for writable files like config).
    In dev mode, this is the project root."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


# Ensure project root is on path
sys.path.insert(0, get_base_path())

from src.config import AppConfig
from gui.app import HairColorApp


def main():
    base_path = get_base_path()
    app_dir = get_app_dir()

    config_path = os.path.join(app_dir, "config.yaml")
    config = AppConfig.load(config_path)

    # Resolve model path relative to bundled resources
    model_path = os.path.join(base_path, config.model_path)
    if not os.path.exists(model_path):
        # Fallback: check relative to exe directory
        model_path = os.path.join(app_dir, config.model_path)

    if not os.path.exists(model_path):
        from tkinter import messagebox
        messagebox.showerror(
            "Model Not Found",
            f"Hair segmentation model not found.\n\n"
            f"Expected at: {model_path}\n\n"
            f"Download it from:\n"
            f"https://storage.googleapis.com/mediapipe-models/"
            f"image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
        )
        sys.exit(1)

    # Update config with resolved path so pipeline uses the correct location
    config.model_path = model_path
    config._save_path = config_path  # Remember where to save config

    app = HairColorApp(config)
    app.mainloop()


if __name__ == "__main__":
    main()
