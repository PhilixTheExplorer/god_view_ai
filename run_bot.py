#!/usr/bin/env python3
"""
Simple script to run the Telegram bot
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.telegram_bot import run_bot

if __name__ == "__main__":
    run_bot()
