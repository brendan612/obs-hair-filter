# OBS Hair Color Filter

Real-time hair color filter for OBS Studio streaming. Uses ML-powered hair segmentation with realistic CIE LAB color space recoloring, running at 40-50+ FPS.

## Features

- **12 built-in color presets** — Jet Black, Dark Brown, Medium Brown, Auburn, Fiery Red, Strawberry Blonde, Platinum Blonde, Ash Blonde, Pastel Pink, Vivid Blue, Emerald Green, Purple
- **Custom color picker** — choose any color via the color dialog
- **Real-time processing** — 40-50+ FPS at 720p with async segmentation
- **Adjustable settings** — intensity, temporal smoothing, and edge softness sliders
- **Live preview** — see the result directly in the app window
- **OBS Virtual Camera output** — appears as a video source in OBS

## Quick Start (Standalone .exe)

1. Download `OBS-Hair-Color-Filter-v1.0.1-win64.zip` from the [Releases](https://github.com/brendan612/obs-hair-filter/releases) page
2. Extract the zip
3. Run `OBS Hair Color Filter.exe`
4. Click **Start** to begin processing
5. In OBS, add a **Video Capture Device** source and select **OBS Virtual Camera**

## Requirements

- **OBS Studio 28+** — start/stop Virtual Camera once to register the device
- **Windows 10/11** (64-bit)
- A webcam

## Development Setup

```bash
# Clone the repo
git clone https://github.com/brendan612/obs-hair-filter.git
cd obs-hair-filter

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the hair segmentation model
curl -L -o models/hair_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite

# Run the app
python run.py
```

### Dependencies

- `opencv-python` — frame capture, color space conversion
- `mediapipe` — ML hair segmentation (DeepLab-v3)
- `pyvirtualcam` — virtual camera output to OBS
- `numpy` — vectorized color math
- `PyYAML` — config persistence
- `Pillow` — GUI image rendering

## Project Structure

```
obs-hair-filter/
├── run.py                    # Entry point
├── requirements.txt
├── config.yaml               # Generated on first run (user settings)
├── models/
│   └── hair_segmenter.tflite # MediaPipe model (downloaded separately)
├── src/
│   ├── capture.py            # Threaded webcam capture (grab/retrieve pattern)
│   ├── segmenter.py          # MediaPipe hair mask with morphological cleanup
│   ├── color_transform.py    # LAB recoloring engine with highlight preservation
│   ├── pipeline.py           # Async segmentation + main processing loop
│   ├── virtual_camera.py     # pyvirtualcam wrapper
│   └── config.py             # Config dataclass + YAML load/save
└── gui/
    └── app.py                # Tkinter GUI (preview, presets, sliders)
```

## How It Works

1. **Capture** — a dedicated thread grabs webcam frames using OpenCV's `grab()`/`retrieve()` pattern to drain driver buffers and minimize latency
2. **Segment** — MediaPipe's hair segmenter runs asynchronously in its own thread, producing a confidence mask. Morphological cleanup and edge-aware bilateral filtering refine the mask edges
3. **Recolor** — the frame is converted to CIE LAB color space. Chrominance channels (a, b) are shifted toward the target color while luminance detail (strand texture, highlights) is preserved. Specular highlights are detected and protected from over-coloring
4. **Output** — the processed frame is sent to OBS Virtual Camera via pyvirtualcam, where OBS picks it up as a Video Capture Device

## Color Presets

| Preset | Style |
|--------|-------|
| Jet Black | Deep black |
| Dark Brown | Natural dark brown |
| Medium Brown | Warm medium brown |
| Auburn | Reddish brown |
| Fiery Red | Vivid red |
| Strawberry Blonde | Light reddish blonde |
| Platinum Blonde | Near-white blonde |
| Ash Blonde | Cool-toned blonde |
| Pastel Pink | Soft pink |
| Vivid Blue | Bright blue |
| Emerald Green | Deep green |
| Purple | Rich purple |
