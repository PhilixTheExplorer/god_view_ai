from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.supabase_client import save_user, get_user_by_id, get_all_users
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message and instructions"""
    welcome_text = """
🌟 Welcome to GodView AI Alert Bot! 🌟

This bot helps you receive real-time alerts from our AI monitoring system.

Available commands:
• /start - Show this welcome message
• /setrole <role> - Set your role (e.g., doctor, nurse, admin)
• /myrole - Check your current role
• /help - Show help information

To get started, set your role with /setrole <your_role>
"""
    await update.message.reply_text(welcome_text)

async def set_role(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set user role"""
    if not context.args:
        await update.message.reply_text(
            "❌ Usage: /setrole <role>\n\n"
            "Examples:\n"
            "• /setrole doctor\n"
            "• /setrole nurse\n"
            "• /setrole admin"
        )
        return
    
    role = context.args[0].lower()
    user_id = str(update.effective_user.id)
    
    try:
        save_user(user_id, role)
        await update.message.reply_text(f"✅ Your role has been set to: {role}")
        logger.info(f"User {user_id} set role to {role}")
    except Exception as e:
        await update.message.reply_text("❌ Failed to save your role. Please try again.")
        logger.error(f"Failed to save user {user_id} role: {e}")

async def my_role(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check current user role"""
    user_id = str(update.effective_user.id)
    
    try:
        user = get_user_by_id(user_id)
        if user:
            await update.message.reply_text(f"👤 Your current role: {user['role']}")
        else:
            await update.message.reply_text(
                "❌ You haven't set a role yet. Use /setrole <role> to set one."
            )
    except Exception as e:
        await update.message.reply_text("❌ Failed to check your role.")
        logger.error(f"Failed to get user {user_id} role: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_text = """
🆘 Help - GodView AI Alert Bot

Commands:
• /start - Welcome message
• /setrole <role> - Set your role for receiving alerts
• /myrole - Check your current role
• /help - Show this help

Roles:
• doctor - Receive medical alerts
• nurse - Receive nursing alerts
• admin - Receive system alerts
• emergency - Receive critical alerts

The bot will automatically send you alerts based on your role when our AI system detects abnormalities.
"""
    await update.message.reply_text(help_text)

def run_bot():
    """Initialize and run the Telegram bot"""
    load_dotenv()
    
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise ValueError("BOT_TOKEN environment variable is required")
    
    logger.info("Starting Telegram bot...")
    logger.info(f"Bot token: {bot_token[:10]}...")
    
    try:
        app = ApplicationBuilder().token(bot_token).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("setrole", set_role))
        app.add_handler(CommandHandler("myrole", my_role))
        app.add_handler(CommandHandler("help", help_command))
        
        logger.info("✅ Bot started successfully. Press Ctrl+C to stop.")
        app.run_polling()
        
    except Exception as e:
        logger.error(f"❌ Bot failed to start: {e}")
        raise

if __name__ == "__main__":
    run_bot()
