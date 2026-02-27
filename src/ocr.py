import easyocr
import cv2
import numpy as np


class PlateOCR:
    def __init__(self, languages: list = ["en"], engine: str = "easyocr"):
        """
        Initialize the OCR reader.

        Args:
            languages: List of language codes e.g. ['en']
            engine: 'easyocr' (default) or 'tesseract'
        """
        self.engine = engine
        self.languages = languages

        if engine == "easyocr":
            self.reader = easyocr.Reader(languages, gpu=False)
        elif engine == "tesseract":
            try:
                import pytesseract
                self.reader = pytesseract
            except ImportError:
                raise ImportError("pytesseract is not installed. Run: pip install pytesseract")

    def preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess cropped plate image to improve OCR accuracy.

        Args:
            plate_crop: Cropped plate region (BGR numpy array)

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Upscale for better OCR (2x)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        # Apply threshold to get clean black/white image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def read(self, plate_crop: np.ndarray) -> str:
        """
        Run OCR on a cropped license plate image.

        Args:
            plate_crop: Cropped plate region (BGR numpy array)

        Returns:
            Detected plate text as string, or empty string if nothing found
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""

        processed = self.preprocess(plate_crop)

        if self.engine == "easyocr":
            results = self.reader.readtext(processed, detail=0, paragraph=False)
            text = " ".join(results).strip().upper()

        elif self.engine == "tesseract":
            config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text = self.reader.image_to_string(processed, config=config).strip().upper()

        else:
            text = ""

        # Clean up: remove non-alphanumeric characters
        cleaned = "".join(c for c in text if c.isalnum() or c == " ").strip()
        return cleaned