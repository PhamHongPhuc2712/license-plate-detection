from ultralytics import YOLO
import numpy as np


class LicensePlateDetector:
    def __init__(self, plate_model_path: str, confidence: float = 0.3, device: str = "cpu"):
        """
        Two-model detector:
        - yolov8n.pt  → detects and tracks cars
        - best.pt     → detects license plates
        """
        self.car_model   = YOLO("yolov8n.pt")
        self.plate_model = YOLO(plate_model_path)
        self.confidence  = confidence
        self.device      = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        1. Detect & track cars
        2. Detect license plates
        3. Assign each plate to its parent car

        Returns list of dicts:
            car_box    : (x1, y1, x2, y2) of the car
            plate_box  : (x1, y1, x2, y2) of the plate
            plate_crop : cropped plate image
            car_id     : tracking ID
            confidence : plate detection confidence
        """
        results = []

        # --- Detect and track cars ---
        car_results = self.car_model.track(
            frame,
            classes=[2, 3, 5, 7],     # car, motorbike, bus, truck
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.5,
            device=self.device,
            verbose=False
        )[0]

        if car_results.boxes.id is None:
            return results

        vehicles = car_results.boxes.data.tolist()  # [x1,y1,x2,y2,track_id,conf,class]

        # --- Detect license plates ---
        plate_results = self.plate_model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )[0]

        # --- Assign each plate to its parent car ---
        for plate in plate_results.boxes.data.tolist():
            px1, py1, px2, py2, score, class_id = plate

            car_box, car_id = self._get_car((px1, py1, px2, py2), vehicles)
            if car_id == -1:
                continue

            cx1, cy1, cx2, cy2 = car_box
            plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]

            results.append({
                "car_box":    (int(cx1), int(cy1), int(cx2), int(cy2)),
                "plate_box":  (int(px1), int(py1), int(px2), int(py2)),
                "plate_crop": plate_crop,
                "car_id":     int(car_id),
                "confidence": round(score, 3)
            })

        return results

    def _get_car(self, plate_box, vehicles):
        """Find which car bounding box contains the plate."""
        px1, py1, px2, py2 = plate_box
        for vehicle in vehicles:
            x1, y1, x2, y2, conf, class_id, car_id = vehicle
            if px1 > x1 and py1 > y1 and px2 < x2 and py2 < y2:
                return (x1, y1, x2, y2), car_id
        return None, -1