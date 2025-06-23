"""
Simple test script to check Telegram bot functionality and user registration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.supabase_client import get_users_by_role, get_all_users, save_user
from dotenv import load_dotenv
import os

load_dotenv()

def check_bot_token():
    """Check if bot token is configured"""
    bot_token = os.getenv("BOT_TOKEN")
    if bot_token:
        print(f"✅ Bot token configured: {bot_token[:10]}...")
        return True
    else:
        print("❌ Bot token not found in environment variables")
        return False

def check_database_connection():
    """Check database connection"""
    try:
        users = get_all_users()
        print(f"✅ Database connected. Total users: {len(users)}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def show_registered_users():
    """Show all registered users"""
    try:
        users = get_all_users()
        if not users:
            print("📋 No users registered yet")
            return
        
        print("📋 Registered Users:")
        for user in users:
            print(f"  - Telegram ID: {user['telegram_id']}, Role: {user['role']}")
        
        # Show users by role
        doctors = get_users_by_role("doctor")
        nurses = get_users_by_role("nurse")
        admins = get_users_by_role("admin")
        
        print(f"\n👨‍⚕️ Doctors: {len(doctors)} users")
        print(f"👩‍⚕️ Nurses: {len(nurses)} users")
        print(f"🔧 Admins: {len(admins)} users")
        
    except Exception as e:
        print(f"❌ Error fetching users: {e}")

def create_test_user():
    """Create a test doctor user for testing"""
    try:
        test_telegram_id = "123456789"  # Test telegram ID
        save_user(test_telegram_id, "doctor")
        print(f"✅ Test doctor user created with ID: {test_telegram_id}")
        print("💡 Note: This is just for testing. Real users should register via the Telegram bot.")
    except Exception as e:
        print(f"❌ Failed to create test user: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 Telegram Bot & Database Check")
    print("=" * 60)
    
    print("\n1. Checking Bot Token...")
    check_bot_token()
    
    print("\n2. Checking Database Connection...")
    check_database_connection()
    
    print("\n3. Current Registered Users:")
    show_registered_users()
    
    print("\n4. Bot Registration Instructions:")
    print("To register users for alerts:")
    print("1. Start the Telegram bot: python src/bot/telegram_bot.py")
    print("2. Users should message the bot and use: /setrole doctor")
    print("3. Once registered, they will receive alerts when created")
    
    print("\n5. Create Test User (optional):")
    response = input("Create a test doctor user? (y/N): ").lower().strip()
    if response == 'y' or response == 'yes':
        create_test_user()
    
    print(f"\n💡 To start the Telegram bot, run:")
    print(f"   python src/bot/telegram_bot.py")
    
    print(f"\n💡 To start the API server, run:")
    print(f"   python start_api.py")

if __name__ == "__main__":
    main()
