import cv2
import csv
import os
import numpy as np
from datetime import datetime


def draw_results(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and plate text on a frame.

    Args:
        frame: Original BGR frame
        results: List of dicts with 'box', 'text', 'confidence'

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for r in results:
        x1, y1, x2, y2 = r["box"]
        text = r.get("text", "")
        conf = r.get("confidence", 0.0)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Build label
        label = f"{text} ({conf:.2f})" if text else f"Plate ({conf:.2f})"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 0), -1)

        # Draw text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 2
        )

    return annotated


def crop_plate(frame: np.ndarray, box: tuple, padding: int = 5) -> np.ndarray:
    """
    Crop the license plate region from the frame with optional padding.

    Args:
        frame: Original BGR frame
        box: (x1, y1, x2, y2) bounding box
        padding: Extra pixels around the box

    Returns:
        Cropped plate image
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # Apply padding safely within frame bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return frame[y1:y2, x1:x2]


def save_crop(crop: np.ndarray, output_dir: str, frame_number: int, plate_index: int) -> str:
    """
    Save a cropped plate image to disk.

    Args:
        crop: Cropped plate image
        output_dir: Directory to save the crop
        frame_number: Current frame number
        plate_index: Index of plate in frame (if multiple)

    Returns:
        Path where the image was saved
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"plate_frame{frame_number:05d}_{plate_index}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, crop)
    return path


def init_csv_log(log_path: str):
    """
    Initialize the CSV log file with headers.

    Args:
        log_path: Full path to the CSV file
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_number", "plate_text", "confidence", "box"])


def append_csv_log(log_path: str, frame_number: int, results: list[dict]):
    """
    Append detection results to the CSV log.

    Args:
        log_path: Full path to the CSV file
        frame_number: Current frame number
        results: Detection results for this frame
    """
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_number,
                r.get("text", ""),
                r.get("confidence", ""),
                str(r.get("box", ""))
            ])


def get_video_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """
    Create a VideoWriter matching the input video's properties.

    Args:
        output_path: Path to save the output video
        cap: Input VideoCapture object (to read FPS and size)

    Returns:
        cv2.VideoWriter object
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))