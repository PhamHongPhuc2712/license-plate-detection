# 🚗 License Plate Detection & OCR Pipeline

An end-to-end pipeline that detects license plates in video footage and extracts their text using deep learning.

## How It Works

The pipeline runs two YOLO models in sequence on every frame, then passes the plate crop to PaddleOCR:

```
Video Frame
  │
  ├─ [1] YOLOv8n (pretrained)  →  Detects & tracks vehicles (car, bus, truck, motorbike)
  │                                 Assigns a persistent car_id via ByteTrack
  │
  ├─ [2] PlateDetectorYolov8n  →  Detects license plate bounding boxes
  │        (custom-trained)         Plates not inside a tracked vehicle are discarded
  │
  ├─ [3] crop_plate()          →  Slices the plate region from the frame (+5px padding)
  │
  └─ [4] PaddleOCR             →  Reads the plate text from the crop
              │
              └─ Outputs annotated video + saved crops + CSV log
```

## Project Structure

```
license-plate-detection/
├── app.py                  # Entry point — loads config, wires components, runs pipeline
├── config/
│   └── config.yaml         # All settings (model path, device, OCR engine, I/O paths)
├── src/
│   ├── detector.py         # LicensePlateDetector — two-model YOLO detection + tracking
│   ├── ocr.py              # PlateOCR — PaddleOCR text extraction
│   ├── pipeline.py         # LicensePlatePipeline — orchestrates detection → OCR → output
│   └── utils.py            # Drawing, cropping, CSV logging, video writer helpers
├── weights/
│   └── PlateDetectorYolov8n.pt  # Custom-trained plate detection model
├── input/
│   └── videos/             # Put your input video here
├── output/
│   ├── videos/             # Annotated output video saved here
│   └── crops/              # Cropped plate images saved here
└── logs/
    └── detections.csv      # Frame-by-frame detection log
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The project requires `paddlepaddle==2.6.2` and `paddleocr==2.7.3` specifically.  
> PaddlePaddle 3.x has an unimplemented oneDNN/PIR kernel bug that breaks CPU inference.

### 3. Add your video

Place your input video at the path specified in `config/config.yaml`:

```yaml
input:
  video_path: "input/videos/test_video.mp4"
```

### 4. Run

```bash
python app.py
```

## Configuration (`config/config.yaml`)

```yaml
model:
  path: "weights/PlateDetectorYolov8n.pt"  # Custom plate detection model
  confidence: 0.5                           # Plate detection threshold (0.0–1.0)
  device: "cuda"                            # "cuda" for GPU, "cpu" for CPU-only

ocr:
  engine: "paddleocr"
  languages:
    - "en"

input:
  video_path: "input/videos/test_video.mp4"

output:
  save_video: true                    # Save annotated output video
  video_path: "output/videos/result.mp4"

  save_crops: true                    # Save cropped plate images per frame
  crops_dir: "output/crops"

  log_csv: true                       # Save detections to CSV
  log_path: "logs/detections.csv"
```

## Output

| Output | Location | Description |
|--------|----------|-------------|
| Annotated video | `output/videos/result.mp4` | Original video with green bounding boxes and plate text labels drawn on each car |
| Plate crops | `output/crops/plate_frame00042_0.jpg` | Cropped plate image per detection (named by frame number and plate index) |
| CSV log | `logs/detections.csv` | Columns: `timestamp`, `frame_number`, `car_id`, `plate_text`, `confidence` |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | latest | YOLOv8 detection + ByteTrack tracking |
| `paddlepaddle` | 2.6.2 | PaddlePaddle inference backend |
| `paddleocr` | 2.7.3 | License plate text recognition |
| `opencv-python` | 4.6.0.66 | Video I/O, image processing, annotation |
| `numpy` | 1.26.4 | Array operations |
| `pyyaml` | latest | Config file parsing |
