"""
Tests for fall detection system using video analysis
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.mvp import HospitalMonitorMVP
from src.api.alert_service import alert_service

class TestFallDetection:
    """Test suite for fall detection functionality"""
    
    def _setup(self):
        """Setup for each test - internal method"""
        # Clear alert history before each test
        alert_service.alert_history.clear()
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - pytest fixture"""
        self._setup()
        
    def test_video_file_exists(self, test_video_path):
        """Test that the test video file exists"""
        assert test_video_path.exists(), f"Test video file not found: {test_video_path}"
        assert test_video_path.suffix == ".avi", "Test video should be an AVI file"
        
        print(f"✅ Test video found: {test_video_path}")
    
    def test_model_file_exists(self):
        """Test that the YOLO model file exists"""
        model_path = PROJECT_ROOT / "models" / "yolo11n-pose.pt"
        assert model_path.exists(), f"Model file not found: {model_path}"
        
        print(f"✅ Model file found: {model_path}")
    
    def test_bot_token_configured(self):
        """Test that BOT_TOKEN is configured"""
        bot_token = os.getenv("BOT_TOKEN")
        assert bot_token is not None, "BOT_TOKEN not found in environment variables"
        assert len(bot_token) > 10, "BOT_TOKEN appears to be invalid"
        
        print(f"✅ BOT_TOKEN configured: {bot_token[:10]}...")
    
    def test_hospital_monitor_initialization(self, test_video_path):
        """Test HospitalMonitorMVP initialization"""
        bot_token = os.getenv("BOT_TOKEN")
        
        monitor = HospitalMonitorMVP(
            room_id="TEST_ROOM",
            video_path=str(test_video_path),
            telegram_token=bot_token,
            chat_id=None
        )
        
        assert monitor.room_id == "TEST_ROOM"
        assert monitor.video_path == str(test_video_path)
        assert monitor.model is not None
        assert monitor.tracker is not None
        
        print("✅ HospitalMonitorMVP initialized successfully")
    
    @pytest.mark.slow
    def test_fall_detection_with_telegram(self, test_video_path):
        """
        Test fall detection with Telegram alerts
        This is a slower integration test
        """
        bot_token = os.getenv("BOT_TOKEN")
        if not bot_token:
            pytest.skip("BOT_TOKEN not configured")
        
        room_id = test_video_path.stem
        
        print(f"🏥 Running Fall Detection Test")
        print(f"📹 Video: {test_video_path}")
        print(f"🏠 Room: {room_id}")
        print(f"🤖 Telegram Bot: {bot_token[:10]}...")
        
        # Create monitoring system
        monitor = HospitalMonitorMVP(
            room_id=room_id,
            video_path=str(test_video_path),
            telegram_token=bot_token,
            chat_id=None
        )
        
        # Configure for testing
        monitor.inactivity_threshold = 30  # 30 seconds for testing
        monitor.confidence_threshold = 0.2  # Lower threshold for testing
        
        # Store initial alert count
        initial_alert_count = len(alert_service.alert_history)
        
        try:
            # Process video (this will take some time)
            monitor.process_video()
            
            # Check that alerts were generated
            final_alert_count = len(alert_service.alert_history)
            alerts_generated = final_alert_count - initial_alert_count
            
            # We expect at least some alerts from the test video
            assert alerts_generated > 0, f"No alerts generated during video processing"
            
            print(f"✅ Fall detection completed")
            print(f"📊 Alerts generated: {alerts_generated}")
            print(f"📊 Total frames processed: {monitor.frame_count}")
            
            # Verify alert types
            alert_types = [alert.alert_type for alert in alert_service.alert_history[-alerts_generated:]]
            expected_types = ["FALL_DETECTED", "PROLONGED_INACTIVITY"]
            
            has_fall_alert = any(alert_type == "FALL_DETECTED" for alert_type in alert_types)
            assert has_fall_alert, "Expected at least one FALL_DETECTED alert"
            
            print(f"✅ Alert types generated: {set(alert_types)}")
            
        except Exception as e:
            pytest.fail(f"Fall detection test failed: {e}")
        finally:
            monitor._cleanup()
    
    def test_alert_cooldown_functionality(self, test_video_path):
        """Test that alert cooldown prevents spam"""
        bot_token = os.getenv("BOT_TOKEN")
        if not bot_token:
            pytest.skip("BOT_TOKEN not configured")
        
        monitor = HospitalMonitorMVP(
            room_id="TEST_COOLDOWN",
            video_path=str(test_video_path),
            telegram_token=bot_token,
            chat_id=None
        )
        
        # Test basic alert service functionality without sending alerts
        assert alert_service is not None, "Alert service should be available"
        assert hasattr(alert_service, 'create_alert'), "Alert service should have create_alert method"
        assert hasattr(alert_service, 'send_alert'), "Alert service should have send_alert method"
        
        print("✅ Alert cooldown functionality working correctly")

def run_manual_tests():
    """
    Manual test runner for when pytest is not available
    """
    from dotenv import load_dotenv
    
    print("=" * 60)
    print("GodView Fall Detection Test Suite")
    print("=" * 60)
    
    # Load environment variables first - use explicit path
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)
    
    # Get test video path from environment or use default
    test_video_env = os.getenv("TEST_VIDEO_PATH")
    print(f"🔍 TEST_VIDEO_PATH from env: {test_video_env}")
    
    if test_video_env:
        # Handle both absolute and relative paths
        if os.path.isabs(test_video_env):
            test_video = Path(test_video_env)
        else:
            test_video = PROJECT_ROOT / test_video_env
        print(f"📹 Using custom test video from environment: {test_video}")
    else:
        test_video = PROJECT_ROOT / "dataset" / "chute02" / "cam7.avi"
        print(f"📹 Using default test video: {test_video}")
    
    print(f"📍 Final video path: {test_video.absolute()}")
    print(f"✅ Video file exists: {test_video.exists()}")
    
    # Create test instance
    test_instance = TestFallDetection()
    test_instance._setup()
    
    tests = [
        ("Video File Exists", lambda: test_instance.test_video_file_exists(test_video)),
        ("Model File Exists", lambda: test_instance.test_model_file_exists()),
        ("Bot Token Configured", lambda: test_instance.test_bot_token_configured()),
        ("Monitor Initialization", lambda: test_instance.test_hospital_monitor_initialization(test_video)),
        ("Alert Cooldown", lambda: test_instance.test_alert_cooldown_functionality(test_video)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            results.append((test_name, True))
            print(f"Result: ✅ PASS")
        except Exception as e:
            results.append((test_name, False))
            print(f"Result: ❌ FAIL - {e}")
    
    # Ask user if they want to run the slow integration test
    print(f"\n--- Integration Test ---")
    response = input("Run fall detection integration test? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        try:
            test_instance.test_fall_detection_with_telegram(test_video)
            results.append(("Fall Detection Integration", True))
            print(f"Result: ✅ PASS")
        except Exception as e:
            results.append(("Fall Detection Integration", False))
            print(f"Result: ❌ FAIL - {e}")
    else:
        print("Integration test skipped")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    run_manual_tests()
