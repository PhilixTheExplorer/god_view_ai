# Enhanced Visual Alert System Documentation

## Overview

The GodView Alert System now features enhanced visual formatting for Telegram alerts with:

- **Color-coded priority levels** (Critical, High, Normal)
- **Medical-specific formatting** for patient alerts
- **Rich visual elements** with emojis and structured layouts
- **HTML formatting** for better readability
- **Role-based customization** for different staff types

## Visual Alert Features

### 🎨 Priority-Based Formatting

#### Critical Priority (🔴⚠️)

- **Color**: Red accents with warning symbols
- **Used for**: Fall detection, cardiac arrest, respiratory distress
- **Visual elements**: Fire emojis, urgent borders, bold text
- **Message**: "CRITICAL - IMMEDIATE ACTION REQUIRED"

#### High Priority (🟠🔥)

- **Color**: Orange accents with fire symbols
- **Used for**: Vital signs abnormal, prolonged inactivity
- **Visual elements**: Lightning bolts, high priority borders
- **Message**: "HIGH PRIORITY - URGENT ATTENTION NEEDED"

#### Normal Priority (🔵📋)

- **Color**: Blue accents with info symbols
- **Used for**: System notifications, routine alerts
- **Visual elements**: Standard borders, informational icons
- **Message**: "STANDARD ALERT"

### 🏥 Medical Alert Formatting

Enhanced patient alerts include:

```
🏥 PATIENT EMERGENCY ALERT 🏥
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤕💥 FALL DETECTED

👤 PATIENT INFORMATION:
├─ 🆔 ID: P-001
├─ 🏠 Room: ICU_001
├─ 🛏️ Bed: B-05
└─ ⏰ Time: 14:30:25

📋 INCIDENT DETAILS:
Patient fall detected - immediate attention required

🎯 Confidence: 95.0%
📹 Frame: #1250

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 IMMEDIATE RESPONSE REQUIRED 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 🔧 System Status Alerts

System notifications include startup, shutdown, errors, and maintenance:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢🚀 SYSTEM STARTUP 🟢🚀
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖥️ System Status Update

📦 Version: v1.0.0
⏱️ Uptime: 2h 15m
🏠 Active Rooms: 8
📹 Connected Cameras: 32
📊 Alerts Count: 15

📅 Timestamp: 14:30:25 - 23/06/2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Alert Type Mapping

### Medical Alerts (Enhanced Formatting)

- `FALL_DETECTED` → 🤕💥 Critical priority
- `PROLONGED_INACTIVITY` → 😴⏰ High priority
- `VITAL_SIGNS_ABNORMAL` → 💓⚠️ Critical priority
- `CARDIAC_ARREST` → 💔🚨 Critical priority
- `RESPIRATORY_DISTRESS` → 🫁⚠️ Critical priority
- `EMERGENCY` → 🚑🆘 Critical priority
- `MEDICATION_DUE` → 💊⏰ High priority

### System Alerts (Standard Formatting)

- `TEST_ALERT` → 🔵📋 Normal priority
- `SYSTEM_ERROR` → ❌🔧 High priority
- `MAINTENANCE` → 🔧⚙️ Normal priority

## Role-Based Icons

- **Doctors**: 👨‍⚕️👩‍⚕️
- **Nurses**: 👩‍⚕️👨‍⚕️
- **Admins**: 🔧👨‍💼
- **Emergency**: 🚑🏥

## HTML Formatting Features

The alerts use Telegram's HTML parsing for:

- `<b>Bold text</b>` for headers and important info
- `<i>Italic text</i>` for descriptions
- `<u>Underlined text</u>` for critical sections
- `<code>Monospace text</code>` for IDs and technical data

## Implementation

### Automatic Detection

The system automatically detects medical vs. system alerts and applies appropriate formatting:

```python
# Medical alerts use enhanced patient formatting
if alert_type in medical_alert_types:
    send_patient_alert(role, alert_type, patient_data)
else:
    send_alert(role, message, priority)
```

### Manual Usage

You can also send formatted alerts directly:

```python
from src.notifications.alert_dispatcher import send_patient_alert

patient_data = {
    "patient_id": "P-001",
    "room_id": "ICU_001",
    "description": "Patient requires immediate attention",
    "confidence": 0.95,
    "frame_number": 1250
}

await send_patient_alert("doctor", "FALL_DETECTED", patient_data)
```

## Testing

### Visual Testing (Offline)

```bash
python test_visual_alerts.py
```

Shows formatted output without sending messages.

### Live Testing

```bash
python test_visual_alerts.py
```

Select live testing option to send actual Telegram messages.

### Demo System

```bash
python demo_alert_system.py
```

Interactive demo with enhanced visual alerts.

## Configuration

### Enable HTML Parsing

The system automatically enables HTML parsing by setting:

```python
data = {
    "chat_id": uid,
    "text": formatted_message,
    "parse_mode": "HTML"
}
```

### Customize Priorities

Modify priority mappings in `alert_dispatcher.py`:

```python
priority_config = {
    "critical": {
        "icon": "🚨🔴",
        "color": "#FF0000",
        "urgency": "CRITICAL - IMMEDIATE ACTION REQUIRED"
    }
    # ...
}
```

## Best Practices

1. **Use appropriate alert types** - Medical alerts get enhanced formatting automatically
2. **Include detailed descriptions** - They're prominently displayed in formatted alerts
3. **Set accurate confidence levels** - Displayed as percentages in patient alerts
4. **Use meaningful patient/room IDs** - They're highlighted in the visual layout
5. **Test formatting changes** - Use the test script before deploying

## Troubleshooting

### HTML Not Rendering

- Ensure `parse_mode: "HTML"` is set in Telegram API calls
- Check for invalid HTML tags in message content
- Verify BOT_TOKEN has necessary permissions

### Visual Elements Not Showing

- Some older Telegram clients may not display all emojis
- Unicode borders may appear differently on various devices
- Test with multiple Telegram clients for consistency

### Performance Considerations

- Enhanced formatting adds ~500ms per alert for processing
- HTML parsing may be slower than plain text
- Consider caching formatted templates for high-volume scenarios
