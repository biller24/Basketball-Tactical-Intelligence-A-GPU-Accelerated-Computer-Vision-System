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

    model = YOLO("models/yolo26m.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=350,
        imgsz=640,         # Higher res for small ball detection
        batch=8,            # 4GB VRAM limit
        patience=40,        # Early stopping: saves time and GPU heat
        optimizer='AdamW',
        lr0=0.001,
        device=0,           # Force use of GPU 1050 Ti
        workers=2,          # Keep low on Windows to avoid 'BrokenPipe' errors
        plots=True,
        name="basketball_v6_yolo11", # Folder name for results
        exist_ok=True,       # Overwrite the folder if  restart
    )

if __name__ == "__main__":
    train_model()