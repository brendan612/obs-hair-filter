import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import AppConfig
from gui.app import HairColorApp


def main():
    config = AppConfig.load("config.yaml")

    # Check model exists
    if not os.path.exists(config.model_path):
        print(f"ERROR: Hair segmentation model not found at '{config.model_path}'")
        print()
        print("Download it with:")
        print(
            f"  curl -L -o {config.model_path} "
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "hair_segmenter/float32/latest/hair_segmenter.tflite"
        )
        sys.exit(1)

    app = HairColorApp(config)
    app.mainloop()


if __name__ == "__main__":
    main()
