# GodView Alert System API

The alert system has been moved from the MVP module to a dedicated API service for better modularity and accessibility.

## Changes Made

### 1. New Alert Service (`src/api/alert_service.py`)

- Moved `AlertSystem` class from `mvp.py` to a dedicated service
- Renamed to `AlertService` for clarity
- Added comprehensive alert management methods
- Improved logging and error handling
- Made the service globally accessible

### 2. Enhanced API (`src/api/main.py`)

- Added Pydantic models for request/response validation
- Added comprehensive alert management endpoints
- Integrated with the new alert service

### 3. Updated MVP Module (`src/ai/mvp.py`)

- Removed local `AlertSystem` class
- Updated to use the global `alert_service`
- Simplified alert creation using `alert_service.create_alert()`
- Maintained all existing functionality

### 4. Telegram Integration

- **Integrated with existing Telegram Alert Dispatcher** - All alerts automatically sent to registered users
- **Role-based alerting** - Alerts sent to users with "doctor" role by default
- **Priority-based messaging** - Supports normal, high, and critical priority levels
- **Uses BOT_TOKEN from environment** - Leverages existing Telegram bot configuration
- **Database integration** - Uses Supabase to manage user roles and Telegram IDs

## API Endpoints

### Health & Info

- `GET /` - Welcome message
- `GET /health` - Health check with alert count

### Alert Management

- `POST /alerts` - Create a new alert
- `GET /alerts` - Get all alerts (with optional limit)
- `GET /alerts/room/{room_id}` - Get alerts for specific room
- `GET /alerts/type/{alert_type}` - Get alerts of specific type
- `GET /alerts/stats` - Get alert statistics
- `DELETE /alerts` - Clear all alert history
- `POST /alerts/test` - Create a test alert

### Legacy

- `POST /predict` - Upload image for detection (existing endpoint)

## Usage

### Starting the API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Creating Alerts via API

```python
import requests

alert_data = {
    "patient_id": 123,
    "room_id": "ROOM_001",
    "alert_type": "FALL_DETECTED",
    "description": "Patient fall detected",
    "confidence": 0.95,
    "frame_number": 1500
}

response = requests.post("http://localhost:8000/alerts", json=alert_data)
```

### Using in MVP Code

```python
from src.api.alert_service import alert_service

# Create and send alert
alert = alert_service.create_alert(
    patient_id=1,
    room_id="ROOM_001",
    alert_type="FALL_DETECTED",
    description="Patient fall detected"
)
alert_service.send_alert(alert)
```

### Testing the API

Run the test suite to verify functionality:

```bash
python test_api.py
```

### Setting Up Telegram Users

1. **Start the Telegram Bot:**

   ```bash
   python src/bot/telegram_bot.py
   ```

2. **Register Users:**

   - Users message the bot and use `/setrole doctor` to register as doctors
   - Use `/setrole nurse` or `/setrole admin` for other roles
   - Check registration with `/myrole`

3. **Check Setup:**
   ```bash
   python check_telegram_setup.py
   ```

## Configuration

The alert service automatically uses the BOT_TOKEN from your .env file and sends alerts to all registered users with the "doctor" role.

```python
# Configure Telegram (optional)
alert_service.telegram_token = "your_bot_token"
alert_service.chat_id = "your_chat_id"
```

## Alert Types

Common alert types used in the system:

- `FALL_DETECTED` - Patient fall detected
- `PROLONGED_INACTIVITY` - Patient inactive for extended period
- `TEST_ALERT` - Test alerts for development

## Benefits of the New Architecture

1. **Modularity**: Alert logic is separated from video processing
2. **API Access**: Alerts can be managed via REST API
3. **Centralized**: Single source of truth for all alerts
4. **Scalable**: Easy to add new alert types and destinations
5. **Testable**: Isolated alert functionality for easier testing
6. **Persistent**: Alert history is maintained across sessions

## File Structure

```
src/
├── api/
│   ├── main.py              # FastAPI application with alert endpoints
│   └── alert_service.py     # Alert service and data models
├── ai/
│   └── mvp.py              # Updated to use alert service
└── ...

start_api.py                 # API startup script
test_api.py                 # API test suite
```
