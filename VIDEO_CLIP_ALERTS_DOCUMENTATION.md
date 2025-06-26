# Video Clip Alert System Documentation

## Overview

The God View monitoring system now sends **short video clips** instead of static snapshots when fall events are detected. This provides much more context and information about the incident.

## Key Features

### **Video Clip Recording**

- **Continuous Buffer**: The system maintains a rolling buffer of the last 5 seconds of video
- **Automatic Capture**: When a fall is detected, the buffered frames are compiled into an MP4 video
- **High Quality**: Videos maintain original frame rate and resolution
- **Compact Size**: Optimized for fast Telegram delivery

### **Fall-Specific Clips**

- **Fall Events**: Video clips show the sequence leading up to and including the fall
- **Context**: Shows patient movement patterns before the incident
- **Duration**: 5-second clips provide sufficient context without being too large
- **Format**: MP4 format compatible with all Telegram clients

### **Hybrid Alert System**

- **Fall Alerts**: Send video clips (more informative)
- **Inactivity Alerts**: Send snapshots (sufficient for static conditions)
- **Fallback**: If video creation fails, system falls back to regular text alerts

## Technical Implementation

### **Video Buffer Management**

```python
# Buffer Configuration
clip_duration_seconds = 5      # 5-second clips
clip_buffer_frames = []        # Rolling frame buffer
fps = 30                       # Frames per second (auto-detected)

# Buffer Update (every frame)
self._update_clip_buffer(frame)

# Maintains buffer size automatically
max_buffer_frames = clip_duration_seconds * fps
```

### **Video Creation Process**

1. **Detection**: Fall event is detected by pose analyzer
2. **Buffer**: Last 5 seconds of frames are retrieved from buffer
3. **Encoding**: Frames are compiled into MP4 video using OpenCV
4. **Verification**: Video file is checked for validity and size
5. **Delivery**: Video is sent via Telegram Bot API

### **Telegram Integration**

```python
# New API Endpoint
BOT_VIDEO_API = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

# Video Alert Function
async def send_video_alert(role, message, video_path, priority)

# Enhanced Alert Service
alert_service.send_video_alert(alert, video_path)
```

## File Structure

### **Video Storage**

```
clips/
├── FALL_DETECTED_cam7_track4_20250626_160015_frame775.mp4
├── FALL_DETECTED_cam7_track4_20250626_160018_frame796.mp4
└── PROLONGED_INACTIVITY_cam7_track4_20250626_160020_frame800.jpg
```

### **Naming Convention**

- **Format**: `{ALERT_TYPE}_{ROOM_ID}_track{TRACK_ID}_{TIMESTAMP}_frame{FRAME_NUMBER}.mp4`
- **Example**: `FALL_DETECTED_cam7_track4_20250626_160015_frame775.mp4`
- **Timestamp**: `YYYYMMDD_HHMMSS` format for easy sorting

## Benefits Over Snapshots

### **📹 Video Clips Advantages**

1. **Context**: Shows what led to the fall
2. **Movement**: Captures the dynamics of the incident
3. **Verification**: Medical staff can see the actual event
4. **Analysis**: Better understanding of fall patterns
5. **Evidence**: More comprehensive documentation

### **📸 Snapshots (Still Used For)**

- Inactivity detection (static state)
- System errors or fallbacks
- Situations where video is not necessary

## Configuration Options

### **Video Settings**

```python
# Clip Duration
clip_duration_seconds = 5      # Adjustable (recommended: 3-10 seconds)

# Video Quality
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
fps = auto_detected                        # Matches source video

# Buffer Management
max_buffer_frames = duration * fps         # Auto-calculated
```

### **Performance Tuning**

```python
# Telegram Upload Timeout
timeout = 30.0  # seconds (videos need longer upload time)

# File Size Optimization
supports_streaming = True  # Enable streaming for large files

# Buffer Memory Management
# Buffer automatically removes old frames to prevent memory issues
```

## Usage Examples

### **Fall Detection with Video**

```
🚨 FALL DETECTED for track 4!
   Posture sequence: ['standing', 'standing', 'lying', 'lying']
🎥 Fall clip saved: clips/FALL_DETECTED_cam7_track4_20250626_160015_frame775.mp4 (150 frames, 5.0s)
✅ Fall alert with video clip sent to Telegram!
```

### **Telegram Alert Message**

```
🚨🔴 HOSPITAL ALERT SYSTEM 🚨🔴
━━━━━━━━━━━━━━━━━━━━

🔥 CRITICAL - IMMEDIATE ACTION REQUIRED

🏥 ALERT DETAILS:
├─ 📋 Type: FALL_DETECTED
├─ 🏠 Room: cam7
├─ 👤 Patient ID: 4
├─ 🎯 Frame: 775
└─ ⏰ Time: 16:00:15

📝 Description:
Patient fall detected in cam7 - immediate attention required.
Fall sequence captured in video clip.

[VIDEO CLIP ATTACHED - 5 seconds showing fall sequence]
```

## Error Handling

### **Video Creation Failures**

- **Insufficient Frames**: Falls back to snapshot if buffer too small
- **Encoding Errors**: Falls back to text-only alert
- **File System Issues**: Graceful degradation to basic alerts
- **Upload Failures**: Retries with timeout handling

### **Fallback Scenarios**

1. **Low Memory**: Automatically manages buffer size
2. **Disk Space**: Monitors available storage
3. **Network Issues**: Handles Telegram API failures
4. **Format Problems**: Validates video files before sending

## Performance Considerations

### **Memory Usage**

- **Buffer Size**: ~150 frames × frame_size (~30MB for 1080p)
- **Auto-Management**: Old frames automatically removed
- **Efficiency**: Only stores frames in memory (no disk writes until needed)

### **Network Usage**

- **File Size**: ~2-5MB per 5-second clip (depending on resolution)
- **Compression**: MP4 provides good compression ratio
- **Streaming**: Telegram supports streaming for large files

### **Processing Speed**

- **Real-time**: Video creation takes ~0.5-1 second
- **Non-blocking**: Video creation doesn't interrupt detection
- **Efficient**: Uses OpenCV optimized encoding

## Testing Recommendations

1. **Test Fall Scenarios**: Verify clips capture full fall sequence
2. **Check Video Quality**: Ensure clips are clear and informative
3. **Validate Telegram Delivery**: Test video upload success rates
4. **Monitor Performance**: Check memory usage and processing speed
5. **Fallback Testing**: Verify graceful degradation when video fails

## Future Enhancements

- **Variable Clip Length**: Adjust duration based on incident type
- **Multiple Angles**: Combine clips from multiple cameras
- **Slow Motion**: Create slow-motion clips for detailed analysis
- **Annotations**: Add overlay information (timestamps, tracking data)
- **Compression Options**: Different quality settings for bandwidth management
