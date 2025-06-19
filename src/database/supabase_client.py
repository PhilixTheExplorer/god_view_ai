from supabase import create_client
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Get environment variables with fallback handling
supabase_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not supabase_url:
    raise ValueError("SUPABASE_URL environment variable is required")
if not supabase_key:
    raise ValueError("SUPABASE_KEY (or SUPABASE_ANON_KEY) environment variable is required")

supabase = create_client(supabase_url, supabase_key)

def get_users_by_role(role: str) -> List[str]:
    """
    Get all user telegram IDs for a specific role
    
    Args:
        role (str): The role to filter by
        
    Returns:
        List[str]: List of telegram IDs
    """
    result = supabase.table("users").select("telegram_id").eq("role", role).execute()
    return [row["telegram_id"] for row in result.data]

def save_user(telegram_id: str, role: str) -> Dict[str, Any]:
    """
    Save or update a user with their role
    
    Args:
        telegram_id (str): Telegram user ID
        role (str): User role
        
    Returns:
        Dict[str, Any]: Supabase response
    """
    return supabase.table("users").upsert({"telegram_id": telegram_id, "role": role}).execute()

def get_all_users() -> List[Dict[str, Any]]:
    """Get all users from the database"""
    result = supabase.table("users").select("*").execute()
    return result.data

def delete_user(telegram_id: str) -> Dict[str, Any]:
    """Delete a user from the database"""
    return supabase.table("users").delete().eq("telegram_id", telegram_id).execute()

def get_user_by_id(telegram_id: str) -> Dict[str, Any]:
    """Get a specific user by telegram ID"""
    result = supabase.table("users").select("*").eq("telegram_id", telegram_id).execute()
    return result.data[0] if result.data else None
