# GodView AI System

A real-time AI-powered monitoring system with YOLOv8 object detection and Telegram notifications.

## Project Structure

```
god_view_model/
├── src/                    # Source code
│   ├── api/               # FastAPI endpoints
│   ├── bot/               # Telegram bot
│   ├── ai/                # AI/ML models (YOLOv8)
│   ├── database/          # Database clients
│   └── notifications/     # Alert system
├── models/                # AI model files
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── assets/                # Static assets
├── main.py               # Main entry point
├── run_bot.py           # Bot runner
└── requirements.txt     # Dependencies
```

## Components

- **FastAPI Server**: Handles image uploads and runs AI detection
- **YOLOv8 Handler**: Processes images for abnormality detection
- **Telegram Bot**: Manages user registration and sends alerts
- **Supabase Integration**: Stores user data and roles
- **Alert Dispatcher**: Sends notifications to registered users

## Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the System**:

   ```bash
   python scripts/start_system.py
   ```

3. **Register with Bot**:

   - Find @GVA_demo_bot on Telegram
   - Send `/start` to begin
   - Send `/setrole doctor` to register as a doctor

4. **Use the API**:
   - Upload images to `http://localhost:8000/predict`
   - View API docs at `http://localhost:8000/docs`

## Manual Startup

If you prefer to start services individually:

```bash
# Start Telegram Bot
python run_bot.py

# Start API Server (in another terminal)
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

Required in `.env` file:

- `BOT_TOKEN`: Telegram bot token
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_ANON_KEY`: Supabase anonymous key

## API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check
- `POST /predict`: Upload image for AI analysis
  - Accepts: Image files
  - Returns: Detected abnormalities and alerts sent

## Bot Commands

- `/start`: Welcome message and instructions
- `/setrole <role>`: Set your role (e.g., doctor, nurse, admin)
- `/myrole`: Check your current role
- `/help`: Show help information

## Available Roles

- `doctor`: Receive medical alerts
- `nurse`: Receive nursing alerts
- `admin`: Receive system alerts
- `emergency`: Receive critical alerts

## Features

- Real-time object detection with YOLOv8
- Automatic alert dispatching to relevant personnel
- Role-based notification system
- RESTful API for image processing
- Enhanced Telegram bot with multiple commands
- Modular architecture with clear separation of concerns
- Comprehensive logging and error handling
