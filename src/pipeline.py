import cv2
import numpy as np

from src.detector import LicensePlateDetector
from src.ocr import PlateOCR
from src.utils import (
    draw_results,
    crop_plate,
    save_crop,
    append_csv_log,
    get_video_writer,
    init_csv_log
)


class LicensePlatePipeline:
    def __init__(self, detector: LicensePlateDetector, ocr: PlateOCR, config: dict):
        """
        Full end-to-end pipeline combining detection + OCR.

        Args:
            detector: LicensePlateDetector instance
            ocr: PlateOCR instance
            config: Full config dict loaded from config.yaml
        """
        self.detector = detector
        self.ocr = ocr
        self.config = config

        self.save_video = config["output"].get("save_video", True)
        self.save_crops = config["output"].get("save_crops", True)
        self.log_csv = config["output"].get("log_csv", True)
        self.skip_frames = config["output"].get("skip_frames", 2)

        self.output_video_path = config["output"].get("video_path", "output/videos/result.mp4")
        self.crops_dir = config["output"].get("crops_dir", "output/crops")
        self.log_path = config["output"].get("log_path", "logs/detections.csv")
        
        # Track history for OCR voting system -> {car_id: [text1, text2, ...]}
        self.track_history = {}

    def _get_best_text(self, car_id: int, new_text: str) -> str:
        """Keep a history of texts for a tracked car and return the most common."""
        if car_id == -1:
            return new_text
            
        if car_id not in self.track_history:
            self.track_history[car_id] = []
            
        # Only add to history if OCR successfully read something
        if new_text:
            self.track_history[car_id].append(new_text)
            
        if not self.track_history[car_id]:
            return ""
            
        from collections import Counter
        # Get the most common string from the tracker
        counts = Counter(self.track_history[car_id])
        return counts.most_common(1)[0][0]

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run detection + OCR on a single frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Tuple of (annotated_frame, list of result dicts)
        """
        detections = self.detector.detect(frame)

        results = []
        for det in detections:
            box = det["plate_box"]
            car_id = det.get("car_id", -1)
            crop = crop_plate(frame, box)
            
            raw_text = self.ocr.read(crop)
            
            # Use voting system to get the most consistent text so far
            best_text = self._get_best_text(car_id, raw_text)
            
            results.append({
                "car_box": det["car_box"],
                "plate_box": box,
                "car_id": car_id,
                "confidence": det["confidence"],
                "text": best_text, # Output the stabilized text
                "crop": crop
            })

        annotated = draw_results(frame, results)
        return annotated, results

    def run_on_video(self):
        """
        Run the full pipeline on a video file.
        Reads from config input path and writes to config output path.
        """
        video_path = self.config["input"]["video_path"]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Setup outputs
        writer = None
        if self.save_video:
            writer = get_video_writer(self.output_video_path, cap)

        if self.log_csv:
            init_csv_log(self.log_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        print(f"Processing video: {video_path} ({total_frames} frames)")

        last_results = []  # reuse for skipped frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for performance on weak hardware
            if frame_number % self.skip_frames == 0:
                annotated, results = self.process_frame(frame)
                last_results = results
            else:
                # Reuse last detections but draw on new frame
                annotated = draw_results(frame, last_results)
                results = last_results

            # Save crops
            if self.save_crops and frame_number % self.skip_frames == 0:
                for i, r in enumerate(results):
                    if r.get("crop") is not None and r["crop"].size > 0:
                        save_crop(r["crop"], self.crops_dir, frame_number, i)

            # Log to CSV
            if self.log_csv and results:
                append_csv_log(self.log_path, frame_number, results)

            # Write to output video
            if writer:
                writer.write(annotated)

            # Print progress every 50 frames
            if frame_number % 50 == 0:
                print(f"  Frame {frame_number}/{total_frames} | Detections: {len(results)}")

            frame_number += 1

        cap.release()
        if writer:
            writer.release()

        print(f"\n✅ Done! Processed {frame_number} frames.")
        if self.save_video:
            print(f"   Output video : {self.output_video_path}")
        if self.save_crops:
            print(f"   Crops saved  : {self.crops_dir}/")
        if self.log_csv:
            print(f"   CSV log      : {self.log_path}")