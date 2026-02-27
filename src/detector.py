from ultralytics import YOLO
import numpy as np


class LicensePlateDetector:
    def __init__(self, model_path: str, confidence: float = 0.5, device: str = "cpu"):
        """
        Initialize the YOLO license plate detector.

        Args:
            model_path: Path to the pre-trained YOLO model (.pt file)
                        or Hugging Face model name
            confidence: Minimum confidence threshold for detections
            device: 'cpu' or 'cuda'
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect license plates in a single frame.

        Args:
            frame: BGR image as numpy array (from OpenCV)

        Returns:
            List of dicts with keys: 'box' (x1,y1,x2,y2), 'confidence'
        """
        results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": round(conf, 3)
                })

        return detections