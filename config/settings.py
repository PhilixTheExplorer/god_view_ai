"""
Configuration settings for GodView AI System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
MODELS_PATH = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config"
ASSETS_PATH = PROJECT_ROOT / "assets"

# Environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model settings
YOLO_MODEL_PATH = MODELS_PATH / "yolov8n.pt"
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))

# Alert settings
ALERT_PRIORITIES = {
    "normal": "🚨",
    "high": "🔥", 
    "critical": "⚠️"
}

ALERT_ROLES = ["doctor", "nurse", "admin", "emergency"]

def validate_config():
    """Validate required configuration"""
    errors = []
    
    if not BOT_TOKEN:
        errors.append("BOT_TOKEN is required")
    
    if not SUPABASE_URL:
        errors.append("SUPABASE_URL is required")
        
    if not SUPABASE_KEY:
        errors.append("SUPABASE_KEY (or SUPABASE_ANON_KEY) is required")
    
    if not YOLO_MODEL_PATH.exists():
        errors.append(f"YOLO model not found at {YOLO_MODEL_PATH}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True
