"""
Test configuration and shared fixtures for GodView tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    """Path to test video file"""
    return PROJECT_ROOT / "dataset" / "chute02" / "cam7.avi"
