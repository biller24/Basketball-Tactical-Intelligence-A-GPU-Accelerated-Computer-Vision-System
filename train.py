import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
import torch


load_dotenv()

# Dynamic Path Resolution
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "datasets" / "Basketball-Players-17" / "data.yaml"

def train_model():
    # We use YOLO11s (Small) because it outperforms YOLOv5l (Large)
    # while using significantly less VRAM.
    model = YOLO("models/yolo11s.pt")

    # We set patience to 50 so it stops if it stops improving
    results = model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=640,
        batch=8,            # 4GB VRAM limit
        patience=50,        # Early stopping: saves time and GPU heat
        optimizer='SGD',    # Great for the "Small Object" (Ball) detection
        lr0=0.01,
        device=0,           # Force use of GPU 1050 Ti
        workers=2,          # Keep low on Windows to avoid 'BrokenPipe' errors
        plots=True,
        name="basketball_v2_yolo11", # Folder name for results
        exist_ok=True,       # Overwrite the folder if  restart
        # --- NEW OPTIMIZATIONS ---
        box = 10.0,  # Care more about box precision (helps small ball)
        cls = 1.0,  # Care more about correct labels (fixes head vs. ball)
        mosaic = 1.0,  # Force multi-object context
        mixup = 0.1  # Prevents overfitting on specific frames
    )

if __name__ == "__main__":
    train_model()