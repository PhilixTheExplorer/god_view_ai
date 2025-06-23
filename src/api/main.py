from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.notifications.alert_dispatcher import send_alert
from src.api.alert_service import alert_service, Alert

app = FastAPI(
    title="GodView AI System API",
    description="Real-time AI-powered monitoring system with YOLOv8 object detection and alert management",
    version="0.1.0"
)

# Pydantic models for request/response
class AlertCreate(BaseModel):
    patient_id: int
    room_id: str
    alert_type: str
    description: str
    confidence: Optional[float] = 0.0
    frame_number: Optional[int] = 0

class AlertResponse(BaseModel):
    patient_id: int
    room_id: str
    alert_type: str
    timestamp: str
    description: str
    confidence: float
    frame_number: int

@app.get("/")
def home():
    return {"message": "Welcome to the GodView AI Alert Bot API. Use /predict to upload an image or /alerts/* for alert management."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "GodView AI API", "alerts_total": len(alert_service.alert_history)}

# Alert Management Endpoints
@app.post("/alerts", response_model=AlertResponse)
async def create_alert(alert_data: AlertCreate):
    """Create a new alert"""
    try:
        alert = alert_service.create_alert(
            patient_id=alert_data.patient_id,
            room_id=alert_data.room_id,
            alert_type=alert_data.alert_type,
            description=alert_data.description,
            confidence=alert_data.confidence,
            frame_number=alert_data.frame_number
        )
        
        success = alert_service.send_alert(alert)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send alert")
        
        return AlertResponse(**alert_service._serialize_alert(alert))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating alert: {str(e)}")

@app.get("/alerts", response_model=List[Dict])
async def get_alerts(limit: int = 100):
    """Get all alerts with optional limit"""
    try:
        alerts = alert_service.get_alert_history(limit=limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.get("/alerts/room/{room_id}", response_model=List[Dict])
async def get_alerts_by_room(room_id: str, limit: int = 100):
    """Get alerts for a specific room"""
    try:
        alerts = alert_service.get_alerts_by_room(room_id, limit=limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts for room: {str(e)}")

@app.get("/alerts/type/{alert_type}", response_model=List[Dict])
async def get_alerts_by_type(alert_type: str, limit: int = 100):
    """Get alerts of a specific type"""
    try:
        alerts = alert_service.get_alerts_by_type(alert_type, limit=limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts by type: {str(e)}")

@app.get("/alerts/stats")
async def get_alert_stats():
    """Get alert statistics"""
    try:
        stats = alert_service.get_alert_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alert stats: {str(e)}")

@app.delete("/alerts")
async def clear_alerts():
    """Clear all alert history"""
    try:
        alert_service.clear_alert_history()
        return {"message": "Alert history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing alerts: {str(e)}")

@app.post("/alerts/test")
async def create_test_alert():
    """Create a test alert for development purposes"""
    try:
        test_alert = alert_service.create_alert(
            patient_id=999,
            room_id="TEST_ROOM",
            alert_type="TEST_ALERT",
            description="This is a test alert created via API",
            confidence=0.95,
            frame_number=1
        )
        
        success = alert_service.send_alert(test_alert)
        
        return {
            "message": "Test alert created",
            "success": success,
            "alert": alert_service._serialize_alert(test_alert)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating test alert: {str(e)}")


