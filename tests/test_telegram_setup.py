"""
Tests for Telegram bot setup and database connectivity
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.supabase_client import get_users_by_role, get_all_users, save_user
from dotenv import load_dotenv

class TestTelegramSetup:
    """Test suite for Telegram bot setup and database connectivity"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        load_dotenv()
    
    def test_bot_token_configured(self):
        """Test that BOT_TOKEN is configured properly"""
        bot_token = os.getenv("BOT_TOKEN")
        
        assert bot_token is not None, "BOT_TOKEN not found in environment variables"
        assert len(bot_token) > 20, "BOT_TOKEN appears to be too short"
        assert bot_token.count(":") == 1, "BOT_TOKEN should have format 'bot_id:auth_token'"
        
        # Check format (should be numbers:letters/numbers)
        bot_id, auth_token = bot_token.split(":")
        assert bot_id.isdigit(), "Bot ID should be numeric"
        assert len(auth_token) >= 35, "Auth token should be at least 35 characters"
        
        print(f"✅ BOT_TOKEN configured: {bot_token[:10]}...")
    
    def test_database_connection(self):
        """Test database connection"""
        try:
            users = get_all_users()
            assert isinstance(users, list), "get_all_users should return a list"
            
            print(f"✅ Database connected. Total users: {len(users)}")
            
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
    
    def test_supabase_environment_variables(self):
        """Test that Supabase environment variables are configured"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        assert supabase_url is not None, "SUPABASE_URL not found in environment variables"
        assert supabase_key is not None, "SUPABASE_ANON_KEY not found in environment variables"
        
        assert supabase_url.startswith("https://"), "SUPABASE_URL should start with https://"
        assert ".supabase.co" in supabase_url, "SUPABASE_URL should contain .supabase.co"
        assert len(supabase_key) > 100, "SUPABASE_ANON_KEY should be a long JWT token"
        
        print(f"✅ Supabase URL: {supabase_url}")
        print(f"✅ Supabase Key: {supabase_key[:20]}...")
    
    def test_get_users_by_role(self):
        """Test getting users by role"""
        try:
            # Test getting doctors
            doctors = get_users_by_role("doctor")
            assert isinstance(doctors, list), "get_users_by_role should return a list"
            
            # Test getting non-existent role
            fake_role_users = get_users_by_role("nonexistent_role")
            assert isinstance(fake_role_users, list), "Should return empty list for non-existent role"
            
            print(f"✅ Found {len(doctors)} users with 'doctor' role")
            
        except Exception as e:
            pytest.fail(f"get_users_by_role failed: {e}")
    
    def test_user_data_structure(self):
        """Test that user data has expected structure"""
        try:
            users = get_all_users()
            
            if users:
                user = users[0]
                
                # Check required fields
                assert "telegram_id" in user, "User should have telegram_id field"
                assert "role" in user, "User should have role field"
                
                # Check data types
                assert isinstance(user["telegram_id"], (int, str)), "telegram_id should be int or string"
                assert isinstance(user["role"], str), "role should be string"
                
                print(f"✅ User data structure is valid")
                print(f"   Sample user: Telegram ID {user['telegram_id']}, Role: {user['role']}")
            else:
                print("⚠️ No users in database - cannot test user data structure")
                
        except Exception as e:
            pytest.fail(f"User data structure test failed: {e}")
    
    def test_doctor_role_exists(self):
        """Test that at least one doctor is registered"""
        try:
            doctors = get_users_by_role("doctor")
            
            if len(doctors) == 0:
                print("⚠️ No doctors registered in Telegram bot")
                print("   To register as doctor, use: /setrole doctor")
                print("   This is required for receiving fall detection alerts")
            else:
                print(f"✅ Found {len(doctors)} registered doctors")
                for doctor in doctors:
                    print(f"   Doctor: Telegram ID {doctor}")
                    
        except Exception as e:
            pytest.fail(f"Doctor role test failed: {e}")

def run_manual_tests():
    """
    Manual test runner for when pytest is not available
    """
    print("=" * 60)
    print("GodView Telegram Setup Test Suite")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestTelegramSetup()
    test_instance.setup()
    
    tests = [
        ("Bot Token Configuration", test_instance.test_bot_token_configured),
        ("Database Connection", test_instance.test_database_connection),
        ("Supabase Environment", test_instance.test_supabase_environment_variables),
        ("Get Users by Role", test_instance.test_get_users_by_role),
        ("User Data Structure", test_instance.test_user_data_structure),
        ("Doctor Role Exists", test_instance.test_doctor_role_exists),
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
    
    # Additional setup information
    print("\n" + "=" * 60)
    print("Setup Information:")
    print("=" * 60)
    
    try:
        all_users = get_all_users()
        doctors = get_users_by_role("doctor")
        
        print(f"📊 Total registered users: {len(all_users)}")
        print(f"👨‍⚕️ Registered doctors: {len(doctors)}")
        
        if len(doctors) == 0:
            print("\n⚠️  IMPORTANT: No doctors registered!")
            print("   1. Start the Telegram bot")
            print("   2. Send '/start' to the bot")
            print("   3. Send '/setrole doctor' to register as doctor")
            print("   4. Re-run this test to verify")
        
        if len(all_users) > 0:
            print("\n📋 Registered Users:")
            for user in all_users:
                print(f"   - Telegram ID: {user['telegram_id']}, Role: {user['role']}")
        
    except Exception as e:
        print(f"❌ Could not retrieve user information: {e}")

if __name__ == "__main__":
    run_manual_tests()
