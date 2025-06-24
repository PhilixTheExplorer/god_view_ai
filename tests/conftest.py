"""
Test configuration and shared fixtures for GodView tests
"""

import pytest
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path, override=True)

@pytest.fixture
def api_base_url():
    """Base URL for API tests"""
    return "http://localhost:8000"

@pytest.fixture
def test_alert_data():
    """Sample alert data for testing"""
    return {
        "patient_id": 123,
        "room_id": "TEST_ROOM_001",
        "alert_type": "FALL_DETECTED",
        "description": "Test fall detection alert",
        "confidence": 0.95,
        "frame_number": 1500
    }

@pytest.fixture
def test_video_path():
    """Path to test video file - reads from TEST_VIDEO_PATH env var"""
    # Get video path from environment variable
    video_path_str = os.getenv("TEST_VIDEO_PATH", "dataset/chute02/cam7.avi")
    
    # Handle both absolute and relative paths
    if Path(video_path_str).is_absolute():
        video_path = Path(video_path_str)
    else:
        video_path = PROJECT_ROOT / video_path_str
    
    print(f"🎬 Using test video: {video_path}")
    print(f"📁 Video exists: {video_path.exists()}")
    
    return video_path
