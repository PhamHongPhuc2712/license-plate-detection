import cv2
import csv
import os
import numpy as np
from datetime import datetime


def draw_results(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Draw box around the CAR and show plate number as label.
    Skips drawing if the OCR engine fails to read the plate text.
    """
    annotated = frame.copy()

    for r in results:
        text = r.get("text", "")
        
        # Skip drawing entirely if no text was read
        if not text:
            continue

        cx1, cy1, cx2, cy2 = r["car_box"]

        # --- Draw box around the CAR ---
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

        # Label background on top of car box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (cx1, cy1 - th - 10), (cx1 + tw + 4, cy1), (0, 255, 0), -1)
        
        # Draw white text
        cv2.putText(annotated, text, (cx1 + 2, cy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return annotated


def crop_plate(frame: np.ndarray, box: tuple, padding: int = 5) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def save_crop(crop: np.ndarray, output_dir: str, frame_number: int, plate_index: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"plate_frame{frame_number:05d}_{plate_index}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, crop)
    return path


def init_csv_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_number", "car_id", "plate_text", "confidence"])


def append_csv_log(log_path: str, frame_number: int, results: list[dict]):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_number,
                r.get("car_id", ""),
                r.get("text", ""),
                r.get("confidence", "")
            ])


def get_video_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))