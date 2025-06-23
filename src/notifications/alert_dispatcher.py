import httpx
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.supabase_client import get_users_by_role
from dotenv import load_dotenv
from typing import List, Optional
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_API = f"https://api.telegram.org/bot{os.getenv('BOT_TOKEN')}/sendMessage"
BOT_PHOTO_API = f"https://api.telegram.org/bot{os.getenv('BOT_TOKEN')}/sendPhoto"

async def send_alert(role: str, message: str, priority: str = "normal") -> bool:
    """
    Send alert message to all users with specified role
    
    Args:
        role (str): Role to send alert to
        message (str): Alert message
        priority (str): Priority level (normal, high, critical)
        
    Returns:
        bool: True if all messages sent successfully
    """
    user_ids = get_users_by_role(role)
    
    if not user_ids:
        logger.warning(f"No users found with role: {role}")
        return False
    
    # Enhanced visual formatting with colors and emojis
    formatted_message = _format_alert_message(message, priority, role)
    
    success_count = 0
    total_users = len(user_ids)
    
    async with httpx.AsyncClient() as client:
        for uid in user_ids:
            try:
                response = await client.post(
                    BOT_API, 
                    data={
                        "chat_id": uid, 
                        "text": formatted_message,
                        "parse_mode": "HTML"  # Enable HTML formatting for colors and styles
                    }
                )
                if response.status_code == 200:
                    success_count += 1
                    logger.info(f"Alert sent successfully to user {uid}")
                else:
                    logger.error(f"Failed to send alert to user {uid}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending alert to user {uid}: {e}")
    
    logger.info(f"Alert sent to {success_count}/{total_users} users")
    return success_count == total_users

def _format_alert_message(message: str, priority: str = "normal", role: str = "doctor") -> str:
    """
    Format alert message with enhanced visual elements, colors, and emojis
    
    Args:
        message (str): Original alert message
        priority (str): Priority level
        role (str): Target role
        
    Returns:
        str: Formatted message with HTML styling
    """
    from datetime import datetime
    
    # Priority-based visual elements
    priority_config = {
        "critical": {
            "icon": "🚨🔴",
            "color": "#FF0000",
            "bg_emoji": "🔥",
            "border": "━━━━━━━━━━━━━━━━━━━━",
            "urgency": "CRITICAL - IMMEDIATE ACTION REQUIRED"
        },
        "high": {
            "icon": "⚠️🟠", 
            "color": "#FF8C00",
            "bg_emoji": "⚡",
            "border": "━━━━━━━━━━━━━━━━━━━━",
            "urgency": "HIGH PRIORITY - URGENT ATTENTION NEEDED"
        },
        "normal": {
            "icon": "🚨🔵",
            "color": "#0066CC", 
            "bg_emoji": "📋",
            "border": "━━━━━━━━━━━━━━━━━━━━",
            "urgency": "STANDARD ALERT"
        }
    }
    
    config = priority_config.get(priority, priority_config["normal"])
    
    # Role-based emojis
    role_emojis = {
        "doctor": "👨‍⚕️👩‍⚕️",
        "nurse": "👩‍⚕️👨‍⚕️", 
        "admin": "🔧👨‍💼",
        "emergency": "🚑🏥"
    }
    
    role_emoji = role_emojis.get(role, "👤")
    timestamp = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
    
    # Parse message to extract structured data if it's from the alert service
    alert_data = _parse_alert_message(message)
    
    if alert_data:
        # Structured alert formatting
        formatted_msg = f"""
{config['border']}
{config['icon']} <b>HOSPITAL ALERT SYSTEM</b> {config['icon']}
{config['border']}

{config['bg_emoji']} <b><u>{config['urgency']}</u></b>

🏥 <b>ALERT DETAILS:</b>
├─ 📋 <b>Type:</b> <code>{alert_data['type']}</code>
├─ 🏠 <b>Room:</b> <code>{alert_data['room']}</code>
├─ 👤 <b>Patient ID:</b> <code>{alert_data['patient_id']}</code>
├─ 🎯 <b>Frame:</b> <code>{alert_data['frame']}</code>
└─ ⏰ <b>Time:</b> <code>{alert_data['time']}</code>

📝 <b>Description:</b>
<i>{alert_data['description']}</i>

{role_emoji} <b>Target:</b> {role.title()} Staff
📅 <b>Received:</b> {timestamp}

{config['border']}
⚡ <b>Action Required - Please Respond Immediately</b> ⚡
{config['border']}
"""
    else:
        # Simple message formatting
        formatted_msg = f"""
{config['border']}
{config['icon']} <b>ALERT - {priority.upper()}</b> {config['icon']}
{config['border']}

{config['bg_emoji']} <b>{config['urgency']}</b>

📝 <b>Message:</b>
<i>{message}</i>

{role_emoji} <b>Target:</b> {role.title()} Staff
📅 <b>Time:</b> {timestamp}

{config['border']}
"""
    
    return formatted_msg.strip()

def _parse_alert_message(message: str) -> dict:
    """
    Parse structured alert message to extract components
    
    Args:
        message (str): Alert message
        
    Returns:
        dict: Parsed alert data or None if not structured
    """
    if "🚨 HOSPITAL ALERT 🚨" in message:
        lines = message.split('\n')
        alert_data = {}
        
        for line in lines:
            if "Type:" in line:
                alert_data['type'] = line.split("Type:")[1].strip()
            elif "Patient ID:" in line:
                alert_data['patient_id'] = line.split("Patient ID:")[1].strip()
            elif "Room:" in line:
                alert_data['room'] = line.split("Room:")[1].strip()
            elif "Frame:" in line:
                alert_data['frame'] = line.split("Frame:")[1].strip()
            elif "Time:" in line:
                alert_data['time'] = line.split("Time:")[1].strip()
            elif "Description:" in line:
                alert_data['description'] = line.split("Description:")[1].strip()
        
        if len(alert_data) >= 3:  # Must have at least 3 fields to be considered structured
            return alert_data
    
    return None

async def send_bulk_alert(roles: List[str], message: str, priority: str = "normal") -> dict:
    """
    Send alert to multiple roles
    
    Args:
        roles (List[str]): List of roles to send alert to
        message (str): Alert message
        priority (str): Priority level
        
    Returns:
        dict: Results summary
    """
    results = {}
    
    for role in roles:
        success = await send_alert(role, message, priority)
        results[role] = success
    
    return results

async def send_system_status(role: str, status_type: str, details: dict) -> bool:
    """
    Send formatted system status updates
    
    Args:
        role (str): Role to send to
        status_type (str): Type of status (startup, shutdown, error, maintenance)
        details (dict): Status details
        
    Returns:
        bool: Success status
    """
    status_message = _format_system_status(status_type, details)
    return await send_alert(role, status_message, "normal")

def _format_system_status(status_type: str, details: dict) -> str:
    """Format system status message with visual elements"""
    from datetime import datetime
    
    status_config = {
        "startup": {
            "icon": "🟢🚀",
            "title": "SYSTEM STARTUP",
            "color": "#00FF00"
        },
        "shutdown": {
            "icon": "🔴⏹️",
            "title": "SYSTEM SHUTDOWN", 
            "color": "#FF0000"
        },
        "error": {
            "icon": "❌🔧",
            "title": "SYSTEM ERROR",
            "color": "#FF4444"
        },
        "maintenance": {
            "icon": "🔧⚙️",
            "title": "MAINTENANCE MODE",
            "color": "#FFA500"
        }
    }
    
    config = status_config.get(status_type, status_config["startup"])
    timestamp = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
    
    status_msg = f"""
━━━━━━━━━━━━━━━━━━━━
{config['icon']} <b>{config['title']}</b> {config['icon']}
━━━━━━━━━━━━━━━━━━━━

🖥️ <b>System Status Update</b>

"""
    
    # Add details
    for key, value in details.items():
        icon = _get_detail_icon(key)
        status_msg += f"{icon} <b>{key.replace('_', ' ').title()}:</b> <code>{value}</code>\n"
    
    status_msg += f"""
📅 <b>Timestamp:</b> {timestamp}

━━━━━━━━━━━━━━━━━━━━
"""
    
    return status_msg.strip()

def _get_detail_icon(key: str) -> str:
    """Get appropriate icon for detail key"""
    icon_map = {
        "version": "📦",
        "uptime": "⏱️",
        "memory": "💾",
        "cpu": "🧠",
        "alerts_count": "📊",
        "active_rooms": "🏠",
        "connected_cameras": "📹",
        "error_message": "💥",
        "service": "⚙️",
        "port": "🌐",
        "database": "🗄️"
    }
    return icon_map.get(key, "📋")

async def send_patient_alert(role: str, alert_type: str, patient_data: dict) -> bool:
    """
    Send specialized patient alert with enhanced formatting
    
    Args:
        role (str): Target role
        alert_type (str): Type of patient alert
        patient_data (dict): Patient and incident data
        
    Returns:
        bool: Success status
    """
    patient_message = _format_patient_alert(alert_type, patient_data)
    priority = _get_alert_priority(alert_type)
    return await send_alert(role, patient_message, priority)

def _format_patient_alert(alert_type: str, data: dict) -> str:
    """Format patient-specific alert with medical context"""
    from datetime import datetime
    
    alert_icons = {
        "FALL_DETECTED": "🤕💥",
        "PROLONGED_INACTIVITY": "😴⏰", 
        "VITAL_SIGNS_ABNORMAL": "💓⚠️",
        "MEDICATION_DUE": "💊⏰",
        "EMERGENCY": "🚑🆘",
        "CARDIAC_ARREST": "💔🚨",
        "RESPIRATORY_DISTRESS": "🫁⚠️"
    }
    
    icon = alert_icons.get(alert_type, "🚨")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    patient_msg = f"""
🏥 <b>PATIENT EMERGENCY ALERT</b> 🏥
━━━━━━━━━━━━━━━━━━━━

{icon} <b><u>{alert_type.replace('_', ' ')}</u></b>

👤 <b>PATIENT INFORMATION:</b>
├─ 🆔 <b>ID:</b> <code>{data.get('patient_id', 'Unknown')}</code>
├─ 🏠 <b>Room:</b> <code>{data.get('room_id', 'Unknown')}</code>
├─ 🛏️ <b>Bed:</b> <code>{data.get('bed_number', 'N/A')}</code>
└─ ⏰ <b>Time:</b> <code>{timestamp}</code>

📋 <b>INCIDENT DETAILS:</b>
<i>{data.get('description', 'No additional details available')}</i>

🎯 <b>Confidence:</b> {data.get('confidence', 0):.1%}
📹 <b>Frame:</b> #{data.get('frame_number', 'N/A')}

━━━━━━━━━━━━━━━━━━━━
🚨 <b>IMMEDIATE RESPONSE REQUIRED</b> 🚨
━━━━━━━━━━━━━━━━━━━━
"""
    
    return patient_msg.strip()

def _get_alert_priority(alert_type: str) -> str:
    """Determine priority based on alert type"""
    critical_alerts = ["FALL_DETECTED", "CARDIAC_ARREST", "RESPIRATORY_DISTRESS", "EMERGENCY"]
    high_priority = ["VITAL_SIGNS_ABNORMAL", "PROLONGED_INACTIVITY"]
    
    if alert_type in critical_alerts:
        return "critical"
    elif alert_type in high_priority:
        return "high"
    else:
        return "normal"

async def send_photo_alert(role: str, message: str, photo_path: str, priority: str = "normal") -> bool:
    """
    Send alert message with photo to all users with specified role
    
    Args:
        role (str): Role to send alert to
        message (str): Alert message
        photo_path (str): Path to the photo file
        priority (str): Priority level (normal, high, critical)
        
    Returns:
        bool: True if all messages sent successfully
    """
    user_ids = get_users_by_role(role)
    
    if not user_ids:
        logger.warning(f"No users found with role: {role}")
        return False
    
    # Check if photo file exists
    if not os.path.exists(photo_path):
        logger.error(f"Photo file not found: {photo_path}")
        return False
    
    # Enhanced visual formatting with colors and emojis
    formatted_message = _format_alert_message(message, priority, role)
    
    success_count = 0
    total_users = len(user_ids)
    
    async with httpx.AsyncClient() as client:
        for uid in user_ids:
            try:
                # Send photo with caption
                with open(photo_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        "chat_id": uid,
                        "caption": formatted_message,
                        "parse_mode": "HTML"
                    }
                    
                    response = await client.post(
                        BOT_PHOTO_API,
                        files=files,
                        data=data
                    )
                    
                if response.status_code == 200:
                    success_count += 1
                    logger.info(f"Photo alert sent successfully to user {uid}")
                else:
                    logger.error(f"Failed to send photo alert to user {uid}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error sending photo alert to user {uid}: {e}")
    
    logger.info(f"Photo alert sent to {success_count}/{total_users} users")
    return success_count == total_users
