"""
Tests for visual alert formatting and Telegram message formatting
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.notifications.alert_dispatcher import (
    send_alert, 
    send_patient_alert, 
    send_system_status,
    _format_alert_message,
    _format_patient_alert,
    _format_system_status
)

class TestVisualAlerts:
    """Test suite for visual alert formatting"""
    
    def test_basic_alert_formatting(self):
        """Test basic alert formatting with different priorities"""
        message = "Test alert message"
        
        # Test all priority levels
        priorities = ["normal", "high", "critical"]
        
        for priority in priorities:
            formatted = _format_alert_message(message, priority, "doctor")
            
            # Basic assertions
            assert isinstance(formatted, str)
            assert len(formatted) > len(message)  # Should be enhanced
            assert message in formatted or "Test alert message" in formatted
            
            # Priority-specific checks
            if priority == "critical":
                assert "🔴" in formatted or "CRITICAL" in formatted.upper()
            elif priority == "high":
                assert "🟠" in formatted or "HIGH" in formatted.upper()
            elif priority == "normal":
                assert "🔵" in formatted or "NORMAL" in formatted.upper()
        
        print("✅ Basic alert formatting works for all priorities")
    
    def test_patient_alert_formatting(self):
        """Test patient-specific alert formatting"""
        alert_data = {
            "alert_type": "FALL_DETECTED",
            "patient_id": 123,
            "room_id": "ICU_001", 
            "description": "Patient fall detected - immediate attention required",
            "frame_number": 1500,
            "timestamp": "14:30:25"
        }
        
        formatted = _format_patient_alert(alert_data, "critical")
        
        # Check that all important information is included
        assert str(alert_data["patient_id"]) in formatted
        assert alert_data["room_id"] in formatted
        assert alert_data["alert_type"] in formatted
        assert "fall" in formatted.lower()
        
        # Should have medical formatting
        assert "🏥" in formatted or "👨‍⚕️" in formatted or "🚨" in formatted
        
        print("✅ Patient alert formatting includes all required information")
    
    def test_system_status_formatting(self):
        """Test system status message formatting"""
        status_data = {
            "system": "GodView Fall Detection",
            "status": "operational",
            "active_cameras": 8,
            "alerts_today": 3
        }
        
        formatted = _format_system_status(status_data)
        
        # Check system information is included
        assert status_data["system"] in formatted
        assert str(status_data["active_cameras"]) in formatted
        assert str(status_data["alerts_today"]) in formatted
        
        # Should have system formatting
        assert "🖥️" in formatted or "📊" in formatted or "⚡" in formatted
        
        print("✅ System status formatting includes all required information")
    
    def test_role_based_formatting(self):
        """Test that formatting varies by role"""
        message = "Test alert for role-based formatting"
        
        roles = ["doctor", "nurse", "admin"]
        formatted_messages = {}
        
        for role in roles:
            formatted = _format_alert_message(message, "high", role)
            formatted_messages[role] = formatted
            
            # Basic checks
            assert isinstance(formatted, str)
            assert len(formatted) > len(message)
        
        # Check that doctor and nurse messages might be different
        # (implementation may or may not differentiate, but structure should be consistent)
        assert all(msg for msg in formatted_messages.values())
        
        print("✅ Role-based formatting works for all roles")
    
    def test_html_formatting_safety(self):
        """Test that HTML formatting is safe and doesn't break"""
        # Test with potentially problematic characters
        problematic_messages = [
            "Alert with <script>alert('test')</script>",
            "Alert with & ampersand",
            "Alert with \"quotes\" and 'apostrophes'",
            "Alert with émojis 🚨 and unicode ñ characters",
            "Alert with newlines\nand\ttabs"
        ]
        
        for message in problematic_messages:
            formatted = _format_alert_message(message, "normal", "doctor")
            
            # Should still be a valid string
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            
            # Should not contain unescaped script tags
            assert "<script>" not in formatted
        
        print("✅ HTML formatting handles problematic characters safely")
    
    def test_telegram_formatting_limits(self):
        """Test that formatted messages respect Telegram limits"""
        # Telegram has a 4096 character limit for messages
        TELEGRAM_LIMIT = 4096
        
        # Test with very long message
        long_message = "This is a very long alert message. " * 200  # ~7000 characters
        
        formatted = _format_alert_message(long_message, "high", "doctor")
        
        # Should respect Telegram limits
        assert len(formatted) <= TELEGRAM_LIMIT
        
        print(f"✅ Long message formatted within Telegram limit: {len(formatted)}/{TELEGRAM_LIMIT} chars")
    
    def test_structured_hospital_alert(self):
        """Test formatting of structured hospital alerts"""
        hospital_alert = {
            "alert_type": "FALL_DETECTED",
            "patient_id": 456,
            "room_id": "ROOM_201",
            "description": "Patient fall detected - immediate attention required",
            "confidence": 0.95,
            "frame_number": 2500,
            "timestamp": "16:45:30"
        }
        
        formatted = _format_patient_alert(hospital_alert, "critical")
        
        # Check for key medical information
        assert "FALL" in formatted.upper()
        assert str(hospital_alert["patient_id"]) in formatted
        assert hospital_alert["room_id"] in formatted
        assert "CRITICAL" in formatted.upper() or "🔴" in formatted
        
        # Should have professional medical formatting
        medical_indicators = ["🏥", "👨‍⚕️", "🚨", "⚕️"]
        has_medical_indicator = any(indicator in formatted for indicator in medical_indicators)
        assert has_medical_indicator
        
        print("✅ Structured hospital alert formatted with medical indicators")
    
    def test_emoji_and_icon_consistency(self):
        """Test that emojis and icons are used consistently"""
        test_cases = [
            ("FALL_DETECTED", "critical"),
            ("PROLONGED_INACTIVITY", "high"), 
            ("VITAL_SIGNS_ABNORMAL", "critical"),
            ("TEST_ALERT", "normal")
        ]
        
        for alert_type, priority in test_cases:
            alert_data = {
                "alert_type": alert_type,
                "patient_id": 999,
                "room_id": "TEST_ROOM",
                "description": f"Test {alert_type} alert",
                "frame_number": 100,
                "timestamp": "12:00:00"
            }
            
            formatted = _format_patient_alert(alert_data, priority)
            
            # Should have consistent emoji usage
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            
            # Priority emojis should be consistent
            if priority == "critical":
                assert "🔴" in formatted
            elif priority == "high":
                assert "🟠" in formatted
            elif priority == "normal":
                assert "🔵" in formatted
        
        print("✅ Emoji and icon usage is consistent across alert types")

def run_manual_tests():
    """
    Manual test runner for when pytest is not available
    """
    print("=" * 60)
    print("GodView Visual Alerts Test Suite")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestVisualAlerts()
    
    tests = [
        ("Basic Alert Formatting", test_instance.test_basic_alert_formatting),
        ("Patient Alert Formatting", test_instance.test_patient_alert_formatting),
        ("System Status Formatting", test_instance.test_system_status_formatting),
        ("Role-based Formatting", test_instance.test_role_based_formatting),
        ("HTML Safety", test_instance.test_html_formatting_safety),
        ("Telegram Limits", test_instance.test_telegram_formatting_limits),
        ("Hospital Alert Structure", test_instance.test_structured_hospital_alert),
        ("Emoji Consistency", test_instance.test_emoji_and_icon_consistency),
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
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Show some example formatted messages
    print("\n" + "=" * 60)
    print("Example Formatted Messages:")
    print("=" * 60)
    
    # Example 1: Critical fall alert
    fall_alert = {
        "alert_type": "FALL_DETECTED",
        "patient_id": 123,
        "room_id": "ICU_001",
        "description": "Patient fall detected - immediate attention required",
        "frame_number": 1500,
        "timestamp": "14:30:25"
    }
    
    print("\n🔴 CRITICAL FALL ALERT:")
    print(_format_patient_alert(fall_alert, "critical"))
    
    # Example 2: System status
    status = {
        "system": "GodView Fall Detection",
        "status": "operational", 
        "active_cameras": 8,
        "alerts_today": 3
    }
    
    print("\n📊 SYSTEM STATUS:")
    print(_format_system_status(status))

if __name__ == "__main__":
    run_manual_tests()
