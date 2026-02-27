import yaml
import os
from src.detector import LicensePlateDetector
from src.ocr import PlateOCR
from src.pipeline import LicensePlatePipeline


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 50)
    print("   License Plate Detection & OCR Pipeline")
    print("=" * 50)

    # ── Load Config ───────────────────────────────
    print("\n[1/4] Loading config...")
    config = load_config("config/config.yaml")
    print(f"      Model   : {config['model']['path']}")
    print(f"      Device  : {config['model']['device']}")
    print(f"      OCR     : {config['ocr']['engine']}")
    print(f"      Video   : {config['input']['video_path']}")

    # ── Load Detector ─────────────────────────────
    print("\n[2/4] Loading YOLO model...")
    detector = LicensePlateDetector(
        model_path=config["model"]["path"],
        confidence=config["model"]["confidence"],
        device=config["model"]["device"]
    )
    print("      ✅ Model loaded!")

    # ── Load OCR ──────────────────────────────────
    print("\n[3/4] Loading OCR engine...")
    ocr = PlateOCR(
        languages=config["ocr"]["languages"],
        engine=config["ocr"]["engine"]
    )
    print(f"      ✅ {config['ocr']['engine']} ready!")

    # ── Run Pipeline ──────────────────────────────
    print("\n[4/4] Running pipeline on video...\n")
    pipeline = LicensePlatePipeline(
        detector=detector,
        ocr=ocr,
        config=config
    )
    pipeline.run_on_video()

    print("\n" + "=" * 50)
    print("   All done! Check your output/ folder.")
    print("=" * 50)


if __name__ == "__main__":
    main()