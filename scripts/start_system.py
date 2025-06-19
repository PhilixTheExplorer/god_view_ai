#!/usr/bin/env python3
"""
Startup script for the GodView AI system
"""
import subprocess
import sys
import os
import time
from pathlib import Path

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import telegram
        import supabase
        import ultralytics
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_file) as f:
        content = f.read()
        
    required_vars = ["BOT_TOKEN", "SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ Environment file is properly configured")
    return True

def start_bot():
    """Start the Telegram bot in background"""
    print("🤖 Starting Telegram bot...")
    bot_process = subprocess.Popen([sys.executable, "-m", "src.bot.telegram_bot"])
    time.sleep(2)  # Give bot time to start
    return bot_process

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])
    return api_process

def main():
    print("🌟 GodView AI System Startup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_requirements():
        return 1
    
    if not check_env_file():
        return 1
    
    try:
        # Start services
        bot_process = start_bot()
        api_process = start_api()
        
        print("\n✅ System started successfully!")
        print("📱 Telegram Bot: @GVA_demo_bot")
        print("🌐 API Server: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for user to stop
        try:
            bot_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            bot_process.terminate()
            api_process.terminate()
            print("✅ All services stopped")
            
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
