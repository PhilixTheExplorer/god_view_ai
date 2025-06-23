from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
import logging
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.notifications.alert_dispatcher import send_alert as send_telegram_alert

@dataclass
class Alert:
    """Alert data structure"""
    patient_id: int
    room_id: str
    alert_type: str
    timestamp: datetime
    description: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    frame_number: int = 0

class AlertService:
    """Alert service for handling and dispatching alerts"""
    
    def __init__(self, telegram_token: str = None, chat_id: str = None):
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.alert_history: List[Alert] = []
        self.alert_cooldown = 30  # 30 seconds between same alerts
        self.logger = logging.getLogger(__name__)
        
        # Default roles to send alerts to
        self.default_alert_roles = ["doctor"]  # Send alerts to doctors by default
        
    def create_alert(self, patient_id: int, room_id: str, alert_type: str, 
                    description: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                    confidence: float = 0.0, frame_number: int = 0) -> Alert:
        """Create a new alert"""
        alert = Alert(
            patient_id=patient_id,
            room_id=room_id,
            alert_type=alert_type,
            timestamp=datetime.now(),
            description=description,
            bbox=bbox,
            confidence=confidence,
            frame_number=frame_number        )
        return alert
        
    def send_alert(self, alert: Alert, roles: List[str] = None) -> bool:
        """Send alert via configured methods"""
        # Check cooldown
        if self._is_in_cooldown(alert):
            self.logger.info(f"Alert {alert.alert_type} for patient {alert.patient_id} is in cooldown")
            return False
            
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.info(f"🚨 ALERT: {alert.alert_type}")
        self.logger.info(f"   Patient ID: {alert.patient_id}")
        self.logger.info(f"   Room: {alert.room_id}")
        self.logger.info(f"   Frame: {alert.frame_number}")
        self.logger.info(f"   Time: {alert.timestamp.strftime('%H:%M:%S')}")
        self.logger.info(f"   Description: {alert.description}")
        
        # Print to console for MVP
        print(f"\n🚨 ALERT: {alert.alert_type}")
        print(f"   Patient ID: {alert.patient_id}")
        print(f"   Room: {alert.room_id}")
        print(f"   Frame: {alert.frame_number}")
        print(f"   Time: {alert.timestamp.strftime('%H:%M:%S')}")
        print(f"   Description: {alert.description}")
        print("-" * 50)
        
        # Send via Telegram to specified roles or default roles
        target_roles = roles or self.default_alert_roles
        success = True
        
        try:
            # Create alert message for Telegram
            message = self._format_alert_message(alert)
            
            # Send to each role using async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            for role in target_roles:
                try:
                    role_success = loop.run_until_complete(
                        send_telegram_alert(role, message, self._get_priority_from_alert_type(alert.alert_type))
                    )
                    if not role_success:
                        self.logger.warning(f"Failed to send alert to role: {role}")
                        success = False
                    else:
                        self.logger.info(f"Alert sent successfully to role: {role}")
                except Exception as e:
                    self.logger.error(f"Error sending alert to role {role}: {e}")
                    success = False
            
            loop.close()
            
        except Exception as e:
            self.logger.error(f"Error in alert sending process: {e}")
            success = False
            
        return success
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for Telegram"""
        return (f"🚨 HOSPITAL ALERT 🚨\n"
                f"Type: {alert.alert_type}\n"
                f"Patient ID: {alert.patient_id}\n"
                f"Room: {alert.room_id}\n"
                f"Frame: {alert.frame_number}\n"
                f"Time: {alert.timestamp.strftime('%H:%M:%S')}\n"
                f"Description: {alert.description}")
    
    def _get_priority_from_alert_type(self, alert_type: str) -> str:
        """Determine priority level based on alert type"""
        priority_mapping = {
            "FALL_DETECTED": "critical",
            "PROLONGED_INACTIVITY": "high", 
            "TEST_ALERT": "normal"
        }
        return priority_mapping.get(alert_type, "normal")
    
    def _is_in_cooldown(self, alert: Alert) -> bool:
        """Check if similar alert was sent recently"""
        cutoff_time = alert.timestamp - timedelta(seconds=self.alert_cooldown)
        
        for prev_alert in self.alert_history:
            if (prev_alert.patient_id == alert.patient_id and
                prev_alert.alert_type == alert.alert_type and
                prev_alert.timestamp > cutoff_time):
                return True
                
        return False
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history as serializable dictionaries"""
        alerts = self.alert_history[-limit:] if limit else self.alert_history
        return [self._serialize_alert(alert) for alert in alerts]
    
    def get_alerts_by_room(self, room_id: str, limit: int = 100) -> List[Dict]:
        """Get alerts for a specific room"""
        room_alerts = [alert for alert in self.alert_history if alert.room_id == room_id]
        alerts = room_alerts[-limit:] if limit else room_alerts
        return [self._serialize_alert(alert) for alert in alerts]
    
    def get_alerts_by_type(self, alert_type: str, limit: int = 100) -> List[Dict]:
        """Get alerts of a specific type"""
        type_alerts = [alert for alert in self.alert_history if alert.alert_type == alert_type]
        alerts = type_alerts[-limit:] if limit else type_alerts
        return [self._serialize_alert(alert) for alert in alerts]
    
    def _serialize_alert(self, alert: Alert) -> Dict:
        """Convert alert to serializable dictionary"""
        alert_dict = asdict(alert)
        alert_dict['timestamp'] = alert.timestamp.isoformat()
        return alert_dict
    
    def clear_alert_history(self):
        """Clear all alert history"""
        self.alert_history.clear()
        self.logger.info("Alert history cleared")
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "alerts_by_type": {},
                "alerts_by_room": {},
                "latest_alert": None
            }
        
        # Count by type
        alerts_by_type = {}
        alerts_by_room = {}
        
        for alert in self.alert_history:
            alerts_by_type[alert.alert_type] = alerts_by_type.get(alert.alert_type, 0) + 1
            alerts_by_room[alert.room_id] = alerts_by_room.get(alert.room_id, 0) + 1
        
        return {
            "total_alerts": len(self.alert_history),
            "alerts_by_type": alerts_by_type,
            "alerts_by_room": alerts_by_room,
            "latest_alert": self._serialize_alert(self.alert_history[-1]) if self.alert_history else None
        }

# Global alert service instance
alert_service = AlertService()
