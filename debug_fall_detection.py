#!/usr/bin/env python3
"""
Debug script to analyze fall detection for chute05/cam7.avi
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ai.mvp import HospitalMonitorMVP

def debug_fall_detection():
    """Run fall detection on the specific video with debug output"""
    
    # Get video path from environment
    video_path = os.getenv("TEST_VIDEO_PATH", "dataset/chute05/cam7.avi")
    
    print(f"🔍 Debugging Fall Detection")
    print(f"📹 Video: {video_path}")
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get bot token
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        print("⚠️ BOT_TOKEN not configured, running without Telegram")
    
    # Create monitoring system with debug enabled
    monitor = HospitalMonitorMVP(
        room_id="chute05_cam7",
        video_path=video_path,
        telegram_token=bot_token,
        chat_id=None
    )
    
    # Ensure debug mode is enabled
    monitor.debug_mode = True
    
    # Lower thresholds for better detection
    monitor.confidence_threshold = 0.2
    monitor.inactivity_threshold = 30  # seconds
    
    print(f"🔧 Configuration:")
    print(f"   Debug mode: {monitor.debug_mode}")
    print(f"   Confidence threshold: {monitor.confidence_threshold}")
    print(f"   Inactivity threshold: {monitor.inactivity_threshold}")
    
    try:
        print(f"\n🎬 Starting video processing...")
        monitor.process_video()
        
        print(f"\n📊 Processing completed!")
        print(f"   Total frames processed: {monitor.frame_count}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if monitor.cap:
            monitor.cap.release()

if __name__ == "__main__":
    debug_fall_detection()
