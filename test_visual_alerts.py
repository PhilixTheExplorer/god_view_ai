"""
Test script for enhanced visual alert formatting
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.notifications.alert_dispatcher import (
    send_alert, 
    send_patient_alert, 
    send_system_status,
    _format_alert_message,
    _format_patient_alert,
    _format_system_status
)

def test_visual_formatting():
    """Test the visual formatting without sending"""
    print("🎨 Testing Enhanced Visual Alert Formatting")
    print("=" * 60)
    
    # Test 1: Basic alert with different priorities
    print("\n1. Basic Alert Formatting:")
    print("-" * 30)
    
    for priority in ["normal", "high", "critical"]:
        message = f"Test {priority} priority alert message"
        formatted = _format_alert_message(message, priority, "doctor")
        print(f"\n{priority.upper()} Priority Alert:")
        print(formatted)
    
    # Test 2: Structured hospital alert
    print("\n\n2. Structured Hospital Alert:")
    print("-" * 30)
    
    hospital_message = """🚨 HOSPITAL ALERT 🚨
Type: FALL_DETECTED
Patient ID: 123
Room: ICU_001
Frame: 1500
Time: 14:30:25
Description: Patient fall detected - immediate attention required"""
    
    formatted_hospital = _format_alert_message(hospital_message, "critical", "doctor")
    print(formatted_hospital)
    
    # Test 3: Patient-specific alert
    print("\n\n3. Patient-Specific Alert:")
    print("-" * 30)
    
    patient_data = {
        "patient_id": "P-001",
        "room_id": "ICU_001", 
        "bed_number": "B-05",
        "description": "Patient shows signs of cardiac distress with irregular heartbeat detected",
        "confidence": 0.94,
        "frame_number": 2750
    }
    
    patient_alert = _format_patient_alert("VITAL_SIGNS_ABNORMAL", patient_data)
    print(patient_alert)
    
    # Test 4: System status
    print("\n\n4. System Status Update:")
    print("-" * 30)
    
    status_details = {
        "version": "v1.0.0",
        "uptime": "2h 15m",
        "active_rooms": "8",
        "connected_cameras": "32",
        "alerts_count": "15"
    }
    
    system_status = _format_system_status("startup", status_details)
    print(system_status)

async def test_live_alerts():
    """Test sending actual alerts (requires registered users)"""
    print("\n\n🚀 Testing Live Alert Sending")
    print("=" * 60)
    
    print("⚠️  This will send actual Telegram messages to registered doctors!")
    response = input("Continue? (y/N): ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("Live testing cancelled.")
        return
    
    # Test different alert types
    test_alerts = [
        {
            "type": "basic",
            "message": "Test alert with enhanced visual formatting",
            "priority": "normal"
        },
        {
            "type": "patient",
            "alert_type": "FALL_DETECTED",
            "data": {
                "patient_id": "TEST-001",
                "room_id": "TEST_ROOM",
                "bed_number": "T-01",
                "description": "Test patient fall alert with full visual formatting",
                "confidence": 0.95,
                "frame_number": 9999
            }
        },
        {
            "type": "system",
            "status_type": "startup",
            "details": {
                "service": "GodView Alert System",
                "version": "v1.0.0-test",
                "port": "8000"
            }
        }
    ]
    
    for i, alert in enumerate(test_alerts, 1):
        print(f"\n📤 Sending test alert {i}/{len(test_alerts)}...")
        
        try:
            if alert["type"] == "basic":
                success = await send_alert("doctor", alert["message"], alert["priority"])
            elif alert["type"] == "patient":
                success = await send_patient_alert("doctor", alert["alert_type"], alert["data"])
            elif alert["type"] == "system":
                success = await send_system_status("doctor", alert["status_type"], alert["details"])
            
            if success:
                print(f"✅ Alert {i} sent successfully")
            else:
                print(f"❌ Alert {i} failed to send")
                
        except Exception as e:
            print(f"❌ Error sending alert {i}: {e}")
    
    print("\n✅ Live alert testing completed!")

def main():
    """Main test function"""
    print("🎨 Enhanced Visual Alert System Test")
    print("=" * 60)
    
    # Test visual formatting (offline)
    test_visual_formatting()
    
    # Ask about live testing
    print("\n" + "=" * 60)
    print("🔴 LIVE TESTING SECTION")
    print("=" * 60)
    
    print("The following tests will send actual Telegram messages.")
    print("Make sure you have:")
    print("• BOT_TOKEN configured")
    print("• Users registered with 'doctor' role")
    print("• Database connection working")
    
    try:
        asyncio.run(test_live_alerts())
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")

if __name__ == "__main__":
    main()
