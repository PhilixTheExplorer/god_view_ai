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
    
    # Format message based on priority
    priority_icons = {
        "normal": "🚨",
        "high": "🔥",
        "critical": "⚠️"
    }
    
    icon = priority_icons.get(priority, "🚨")
    formatted_message = f"{icon} ALERT - {priority.upper()}\n\n{message}"
    
    success_count = 0
    total_users = len(user_ids)
    
    async with httpx.AsyncClient() as client:
        for uid in user_ids:
            try:
                response = await client.post(
                    BOT_API, 
                    data={"chat_id": uid, "text": formatted_message}
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
