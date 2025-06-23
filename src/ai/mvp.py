from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from collections import defaultdict, deque
import argparse
import os
import math
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.alert_service import alert_service, Alert

# Get the path to the models directory
models_dir = Path(__file__).parent.parent.parent / "models"
model_path = models_dir / "yolo11n-pose.pt"

# Initialize YOLO model
model = YOLO(str(model_path))

@dataclass
class PoseDetection:
    """Patient pose detection data structure"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    timestamp: datetime
    keypoints: np.ndarray  # 17x3 array (x, y, confidence) for COCO pose
    posture: str
    floor_proximity: float
    pose_confidence: float

class PoseAnalyzer:
    """Analyzes patient poses using keypoints and detects anomalies"""
    
    def __init__(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.floor_threshold = 0.80  # 80% down the frame is considered floor level
        
        # COCO pose keypoint indices
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
    def analyze_pose(self, bbox: Tuple[int, int, int, int], 
                    keypoints: np.ndarray) -> Tuple[str, float, float]:
        """Analyze posture from keypoints and bounding box"""
        x1, y1, x2, y2 = bbox
        floor_proximity = y2 / self.frame_height
        
        # Calculate pose confidence (average of visible keypoints)
        valid_keypoints = keypoints[keypoints[:, 2] > 0.3]  # confidence > 0.3
        pose_confidence = np.mean(valid_keypoints[:, 2]) if len(valid_keypoints) > 0 else 0.0
        
        # Analyze posture based on keypoints
        posture = self._classify_posture(keypoints, bbox)
        
        return posture, floor_proximity, pose_confidence
    
    def _classify_posture(self, keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Classify posture based on keypoint positions"""
        x1, y1, x2, y2 = bbox
        
        # Get key body parts
        head_y = self._get_keypoint_y(keypoints, ['nose', 'left_eye', 'right_eye'])
        shoulder_y = self._get_keypoint_y(keypoints, ['left_shoulder', 'right_shoulder'])
        hip_y = self._get_keypoint_y(keypoints, ['left_hip', 'right_hip'])
        knee_y = self._get_keypoint_y(keypoints, ['left_knee', 'right_knee'])
        ankle_y = self._get_keypoint_y(keypoints, ['left_ankle', 'right_ankle'])
        
        # Calculate body orientation
        if head_y is None or hip_y is None:
            return "unknown"
            
        # Vertical alignment check
        vertical_span = hip_y - head_y if head_y < hip_y else 0
        bbox_height = y2 - y1
        
        # Check if person is upright
        if vertical_span > bbox_height * 0.6:  # Significant vertical span
            if ankle_y and ankle_y > hip_y:  # Feet below hips
                return "standing"
            else:
                return "sitting"
        
        # Check for lying down
        if shoulder_y and hip_y:
            # If shoulders and hips are roughly at the same level (horizontal)
            if abs(shoulder_y - hip_y) < bbox_height * 0.2:
                return "lying"
        
        # Check floor proximity for fallen person
        if (y2 / self.frame_height) > self.floor_threshold:
            return "lying"
            
        return "unknown"
    
    def _get_keypoint_y(self, keypoints: np.ndarray, joint_names: List[str]) -> Optional[float]:
        """Get average Y coordinate of specified joints"""
        y_coords = []
        for joint_name in joint_names:
            if joint_name in self.KEYPOINTS:
                idx = self.KEYPOINTS[joint_name]
                if idx < len(keypoints) and keypoints[idx, 2] > 0.3:  # confidence > 0.3
                    y_coords.append(keypoints[idx, 1])
        
        return np.mean(y_coords) if y_coords else None
    def detect_fall(self, track_history: deque) -> bool:
        """Detect fall based on posture changes over time - Enhanced for testing"""
        if len(track_history) < 2:  # Reduced minimum history
            return False
            
        recent = list(track_history)[-8:]  # Look at last 8 detections
        
        # Look for transition from standing/sitting to lying
        standing_sitting_count = 0
        lying_count = 0
        unknown_count = 0
        
        for detection in recent:
            if detection.posture in ["standing", "sitting"]:
                standing_sitting_count += 1
            elif detection.posture == "lying":
                lying_count += 1
            else:
                unknown_count += 1
        
        # Enhanced fall detection conditions
        # Condition 1: Clear transition to lying position
        if lying_count >= 2 and standing_sitting_count >= 1:
            recent_lying = [d for d in recent if d.posture == "lying"]
            if recent_lying and recent_lying[-1].floor_proximity > 0.7:  # Lowered threshold
                print(f"   └─ Fall detected: lying_count={lying_count}, standing_sitting_count={standing_sitting_count}")
                return True
        
        # Condition 2: High floor proximity with posture change
        if len(recent) >= 3:
            latest_detection = recent[-1]
            if (latest_detection.floor_proximity > 0.8 and 
                latest_detection.posture in ["lying", "unknown"]):
                # Check if there was a position change
                earlier_detections = recent[:-2]
                if any(d.floor_proximity < 0.6 for d in earlier_detections):
                    print(f"   └─ Fall detected: high floor proximity with position change")
                    return True
        
        # Condition 3: Rapid position change (bbox vertical movement)
        if len(recent) >= 3:
            positions = [self._get_bbox_center(d.bbox) for d in recent]
            vertical_movements = []
            for i in range(1, len(positions)):
                vertical_movement = positions[i][1] - positions[i-1][1]
                vertical_movements.append(vertical_movement)
            
            # Check for significant downward movement
            if any(movement > 50 for movement in vertical_movements):  # Pixels moved down
                print(f"   └─ Fall detected: rapid downward movement")
                return True
                
        return False
    
    def detect_prolonged_inactivity(self, track_history: deque, 
                                  threshold_seconds: int = 300) -> bool:
        """Detect prolonged inactivity (5 minutes default)"""
        if len(track_history) < 10:
            return False
            
        recent = list(track_history)
        if not recent:
            return False
            
        # Check time span
        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span.total_seconds() < threshold_seconds:
            return False
            
        # Check for minimal movement
        first_center = self._get_bbox_center(recent[0].bbox)
        movement_threshold = 30  # pixels
        
        for detection in recent[1:]:
            curr_center = self._get_bbox_center(detection.bbox)
            distance = math.sqrt((first_center[0] - curr_center[0])**2 + 
                               (first_center[1] - curr_center[1])**2)
            
            if distance > movement_threshold:
                return False
                
        return True
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class SimpleTracker:
    """Simple tracker for MVP - tracks people across frames"""
    
    def __init__(self, max_age: int = 30):
        self.tracks: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_age))
        self.last_seen: Dict[int, datetime] = {}
        self.next_id = 1
        
    def update(self, detections: List[PoseDetection]) -> List[PoseDetection]:
        """Update tracks with new detections"""
        tracked_detections = []
        
        for detection in detections:
            # Simple tracking based on bbox overlap
            best_id = self._find_best_match(detection)
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            
            detection.id = best_id
            self.tracks[best_id].append(detection)
            self.last_seen[best_id] = detection.timestamp
            tracked_detections.append(detection)
            
        return tracked_detections
    
    def _find_best_match(self, detection: PoseDetection) -> Optional[int]:
        """Find best matching track for detection"""
        best_id = None
        max_overlap = 0.3  # Minimum overlap threshold
        
        det_bbox = detection.bbox
        
        for track_id, track_history in self.tracks.items():
            if not track_history:
                continue
                
            last_detection = track_history[-1]
            overlap = self._calculate_overlap(det_bbox, last_detection.bbox)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_id = track_id
                
        return best_id
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU overlap between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class HospitalMonitorMVP:
    """MVP Hospital Monitoring System for video analysis"""
    
    def __init__(self, room_id: str, video_path: str, 
                 telegram_token: str = None, chat_id: str = None):
        self.room_id = room_id
        self.video_path = video_path
        self.running = False
        
        # Initialize components
        self.tracker = SimpleTracker()
        
        # Configure global alert service with telegram credentials
        if telegram_token:
            alert_service.telegram_token = telegram_token
        if chat_id:
            alert_service.chat_id = chat_id
        
        self.model = model
            
        # Video capture
        self.cap = None
        self.pose_analyzer = None
          # Configuration
        self.inactivity_threshold = 10  # Reduced for testing
        self.confidence_threshold = 0.3
        self.frame_count = 0
        
        # Fall detection sensitivity for testing
        self.fall_detection_enabled = True
        self.debug_mode = True  # Enable debug logging
        
        # Create snapshots directory
        self.snapshots_dir = Path("snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)
    
    def _save_snapshot(self, frame: np.ndarray, alert_type: str, track_id: int) -> str:
        """Save a snapshot of the current frame for alerts"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{alert_type}_{self.room_id}_track{track_id}_{timestamp}_frame{self.frame_count}.jpg"
            filepath = self.snapshots_dir / filename
            
            # Save the frame
            cv2.imwrite(str(filepath), frame)
            
            # Log the snapshot
            print(f"📸 Snapshot saved: {filepath}")
            
            return str(filepath)
        except Exception as e:
            print(f"❌ Error saving snapshot: {e}")
            return None
    
    def process_video(self):
        """Process video file for patient monitoring"""

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Failed to open video: {self.video_path}")
            return
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame delay to maintain original video timing
        frame_delay = int(1000 / fps) if fps > 0 else 33  # milliseconds per frame
        
        print(f"Processing video: {self.video_path}")
        print(f"FPS: {fps:.2f}, Duration: {duration:.2f}s, Total frames: {total_frames}")
        print(f"Frame delay: {frame_delay}ms (to maintain original speed)")
        
        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to read first frame")
            return
            
        height, width = frame.shape[:2]
        self.pose_analyzer = PoseAnalyzer(height, width)
        
        print(f"Frame size: {width}x{height}")
        print("Starting analysis... Press 'q' to quit, 'p' to pause")
        
        self.running = True
        self.frame_count = 0
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached")
                break
                
            self.frame_count += 1
            
            try:                # Process frame
                detections = self._detect_poses(frame)
                tracked_detections = self.tracker.update(detections)
                self._analyze_anomalies(frame)
                  # Display frame
                self._display_frame(frame, tracked_detections)
                
                # Control playback speed - maintain original video timing
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)  # Pause until any key
                    
            except Exception as e:
                print(f"Error processing frame {self.frame_count}: {e}")
                continue
        
        self._cleanup()
    
    def _detect_poses(self, frame: np.ndarray) -> List[PoseDetection]:
        """Detect people and their poses using YOLOv8n-pose"""
        detections = []
        
        if self.model is None:
            return detections
            
        try:
            # Run pose detection
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and keypoints is not None:
                    for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                        confidence = float(box.conf)
                        if confidence < self.confidence_threshold:
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Get keypoints (17 keypoints for COCO pose)
                        kpt_array = kpts.data[0].cpu().numpy()  # Shape: (17, 3)
                        
                        # Analyze pose
                        posture, floor_prox, pose_conf = self.pose_analyzer.analyze_pose(
                            (x1, y1, x2, y2), kpt_array)
                        
                        detection = PoseDetection(
                            id=0,  # Will be assigned by tracker
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            timestamp=datetime.now(),
                            keypoints=kpt_array,
                            posture=posture,
                            floor_proximity=floor_prox,
                            pose_confidence=pose_conf                        )
                        
                        detections.append(detection)
                        
        except Exception as e:
            print(f"Error in pose detection: {e}")
            
        return detections
    
    def _analyze_anomalies(self, frame: np.ndarray):
        """Analyze tracks for anomalies and generate alerts"""
        current_time = datetime.now()
        
        for track_id, track_history in self.tracker.tracks.items():
            if not track_history:
                continue
              # Debug logging for fall detection
            if self.debug_mode and len(track_history) >= 3:
                recent_postures = [d.posture for d in list(track_history)[-5:]]
                print(f"Track {track_id}: Recent postures: {recent_postures}")
            
            # Check for falls
            if self.pose_analyzer.detect_fall(track_history):
                print(f"🚨 FALL DETECTED for track {track_id}!")
                
                # Save snapshot for fall detection
                snapshot_path = self._save_snapshot(frame, "FALL_DETECTED", track_id)
                
                alert = alert_service.create_alert(
                    patient_id=track_id,
                    room_id=self.room_id,
                    alert_type="FALL_DETECTED",
                    description=f"Patient fall detected in {self.room_id} - immediate attention required. Analysis shows transition from upright to lying position.",
                    bbox=track_history[-1].bbox,
                    confidence=track_history[-1].confidence,
                    frame_number=self.frame_count                )
                
                if snapshot_path:
                    # Send alert with photo
                    success = alert_service.send_photo_alert(alert, snapshot_path)
                    if success:
                        print(f"✅ Fall alert with photo sent to Telegram!")
                    else:
                        print(f"❌ Failed to send fall alert with photo to Telegram")
                else:
                    # Fallback to regular alert if snapshot failed
                    success = alert_service.send_alert(alert)
                    if success:
                        print(f"✅ Fall alert sent to Telegram!")
                    else:
                        print(f"❌ Failed to send fall alert to Telegram")
            
            # Check for prolonged inactivity
            if self.pose_analyzer.detect_prolonged_inactivity(
                track_history, self.inactivity_threshold):
                print(f"⏰ PROLONGED INACTIVITY detected for track {track_id}")
                
                # Save snapshot for inactivity detection
                snapshot_path = self._save_snapshot(frame, "PROLONGED_INACTIVITY", track_id)
                
                alert = alert_service.create_alert(
                    patient_id=track_id,
                    room_id=self.room_id,
                    alert_type="PROLONGED_INACTIVITY",
                    description=f"Patient in {self.room_id} has been inactive for over {self.inactivity_threshold} seconds. No significant movement detected.",
                    bbox=track_history[-1].bbox,
                    confidence=track_history[-1].confidence,
                    frame_number=self.frame_count
                )
                
                if snapshot_path:
                    # Send alert with photo
                    success = alert_service.send_photo_alert(alert, snapshot_path)
                    if success:
                        print(f"✅ Inactivity alert with photo sent to Telegram!")
                    else:
                        print(f"❌ Failed to send inactivity alert with photo to Telegram")
                else:
                    # Fallback to regular alert if snapshot failed
                    success = alert_service.send_alert(alert)
                    if success:
                        print(f"✅ Inactivity alert sent to Telegram!")
                    else:
                        print(f"❌ Failed to send inactivity alert to Telegram")
    
    def _display_frame(self, frame: np.ndarray, detections: List[PoseDetection]):
        """Display frame with pose overlays"""
        display_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on posture
            color_map = {
                'standing': (0, 255, 0),    # Green
                'sitting': (255, 255, 0),   # Yellow  
                'lying': (0, 0, 255),       # Red
                'unknown': (128, 128, 128)  # Gray
            }
            color = color_map.get(detection.posture, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints
            self._draw_keypoints(display_frame, detection.keypoints, color)
            
            # Draw info text
            info_text = f"ID:{detection.id} {detection.posture}"
            cv2.putText(display_frame, info_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
          # Add system info
        info_text = f"Frame: {self.frame_count} | Room: {self.room_id} | Tracks: {len(self.tracker.tracks)}"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add alerts count
        alert_text = f"Total Alerts: {len(alert_service.alert_history)}"
        cv2.putText(display_frame, alert_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(f"Hospital Monitor MVP - {self.room_id}", display_frame)
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]):
        """Draw pose keypoints on frame"""
        # COCO pose connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3):
                
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                cv2.line(frame, start_point, end_point, color, 2)
    
    def _cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
          # Print summary
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total alerts generated: {len(alert_service.alert_history)}")
        
        # Print alert summary
        if alert_service.alert_history:
            print("\nAlert Summary:")
            for alert in alert_service.alert_history:
                print(f"  - {alert.alert_type} (Patient {alert.patient_id}) at frame {alert.frame_number}")

def main():
    """Main function for MVP"""
    parser = argparse.ArgumentParser(description='Hospital Patient Monitoring System MVP')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--room-id', default='ROOM_001', help='Room identifier')
    parser.add_argument('--telegram-token', help='Telegram bot token for alerts')
    parser.add_argument('--chat-id', help='Telegram chat ID for alerts')
    parser.add_argument('--inactivity-threshold', type=int, default=10, 
                       help='Inactivity threshold in seconds')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print("Hospital Patient Monitoring System MVP")
    print("=" * 50)
    
    # Create and run monitoring system
    monitor = HospitalMonitorMVP(
        room_id=args.room_id,
        video_path=args.video,
        telegram_token=args.telegram_token,
        chat_id=args.chat_id
    )
    
    monitor.inactivity_threshold = args.inactivity_threshold
    
    try:
        monitor.process_video()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor._cleanup()

if __name__ == "__main__":
    main()