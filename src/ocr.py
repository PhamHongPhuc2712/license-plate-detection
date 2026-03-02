import numpy as np
from paddleocr import PaddleOCR

class PlateOCR:
    def __init__(self, languages: list = ["en"], engine: str = "paddleocr"):
        self.engine = engine
        self.languages = languages
        self.reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

    def read(self, plate_crop: np.ndarray) -> str:
        """
        Run OCR on a cropped license plate image using PaddleOCR.
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""

        try:
            result = self.reader.ocr(plate_crop, cls=False)

            if not result or not result[0]:
                return ""

            box, (detected_text, confidence) = result[0][0]
            raw_text = detected_text.upper()
            cleaned = "".join(c for c in raw_text if c.isalnum() or c == " ").strip()
            return cleaned

        except Exception as e:
            return ""