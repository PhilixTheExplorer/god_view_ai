"""
Quick Start Guide for GodView Alert System
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.alert_service import alert_service

def demo_alert_creation():
    """Demonstrate creating alerts programmatically"""
    print("🚨 Creating Sample Alerts...")
    
    # Create different types of alerts
    alerts_to_create = [
        {
            "patient_id": 101,
            "room_id": "ICU_001",
            "alert_type": "FALL_DETECTED", 
            "description": "Patient fall detected in ICU - immediate attention required"
        },
        {
            "patient_id": 102,
            "room_id": "WARD_A_205",
            "alert_type": "PROLONGED_INACTIVITY",
            "description": "Patient has been inactive for over 30 minutes"
        },
        {
            "patient_id": 103,
            "room_id": "EMERGENCY_001",
            "alert_type": "VITAL_SIGNS_ABNORMAL",
            "description": "Abnormal vital signs detected - heart rate irregularity"
        }
    ]
    
    print(f"📤 Sending {len(alerts_to_create)} alerts to all doctors via Telegram...")
    
    for i, alert_data in enumerate(alerts_to_create, 1):
        print(f"\n🔄 Creating alert {i}/{len(alerts_to_create)}:")
        print(f"   Patient: {alert_data['patient_id']}")
        print(f"   Room: {alert_data['room_id']}")
        print(f"   Type: {alert_data['alert_type']}")
        
        # Create alert
        alert = alert_service.create_alert(**alert_data)
        
        # Send alert (this will go to all doctors via Telegram)
        success = alert_service.send_alert(alert)
        
        if success:
            print(f"   ✅ Alert sent successfully")
        else:
            print(f"   ❌ Alert sending failed")
    
    print(f"\n📊 Total alerts in system: {len(alert_service.alert_history)}")

def show_alert_stats():
    """Show current alert statistics"""
    print("\n📈 Alert Statistics:")
    stats = alert_service.get_alert_stats()
    
    print(f"   Total Alerts: {stats['total_alerts']}")
    
    if stats['alerts_by_type']:
        print("   By Type:")
        for alert_type, count in stats['alerts_by_type'].items():
            print(f"     - {alert_type}: {count}")
    
    if stats['alerts_by_room']:
        print("   By Room:")
        for room, count in stats['alerts_by_room'].items():
            print(f"     - {room}: {count}")

def main():
    """Main demonstration"""
    print("=" * 60)
    print("🏥 GodView Alert System - Quick Start Demo")
    print("=" * 60)
    
    print("\n💡 This demo shows how alerts are created and sent to Telegram users.")
    print("💡 Make sure you have:")
    print("   1. BOT_TOKEN configured in .env")
    print("   2. Users registered as 'doctor' role via Telegram bot")
    print("   3. Database connection working")
    
    response = input(f"\n🚀 Ready to send demo alerts to all doctors? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        demo_alert_creation()
        show_alert_stats()
        
        print(f"\n✅ Demo completed!")
        print(f"💬 Check your Telegram - doctors should have received the alerts!")
        
    else:
        print("Demo cancelled. You can run this script anytime to test the system.")
    
    print(f"\n📚 Next Steps:")
    print(f"   • Start API server: python start_api.py")
    print(f"   • Test API: python test_api.py") 
    print(f"   • Run MVP with video: python src/ai/mvp.py --video path/to/video.mp4")
    print(f"   • Check Telegram setup: python check_telegram_setup.py")

if __name__ == "__main__":
    main()
