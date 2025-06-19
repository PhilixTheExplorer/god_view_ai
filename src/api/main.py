from fastapi import FastAPI, File, UploadFile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ai.yolov8_handler import detect_abnormalities
from src.notifications.alert_dispatcher import send_alert

app = FastAPI(
    title="GodView AI System API",
    description="Real-time AI-powered monitoring system with YOLOv8 object detection",
    version="0.1.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to the GodView AI Alert Bot API. Use /predict to upload an image."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "GodView AI API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # alerts = detect_abnormalities(temp_path)
    alerts = ["Example alert: Abnormality detected in image." for x in range(2)]  # Placeholder for actual detection logic
    os.remove(temp_path)

    for alert in alerts:
        await send_alert("doctor", f"Detected abnormality: {alert}")

    return {"status": "ok", "alerts": alerts}


