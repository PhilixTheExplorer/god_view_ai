from ultralytics import YOLO
import os
from pathlib import Path

# Get the path to the models directory
models_dir = Path(__file__).parent.parent.parent / "models"
model_path = models_dir / "yolov8n-pose.pt"

# Initialize YOLO model
model = YOLO(str(model_path))

def detect_abnormalities(image_path):
    """
    Detect abnormalities in an image using YOLOv8
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: List of detected abnormalities
    """
    results = model(image_path)
    alerts = []

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            if cls in ["fall", "lying", "collapse"]:
                alerts.append(cls)

    return alerts

def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_path": str(model_path),
        "model_type": "YOLOv8n",
        "classes": list(model.names.values())
    }
