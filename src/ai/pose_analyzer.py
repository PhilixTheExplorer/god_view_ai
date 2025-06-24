import numpy as np
import math
from typing import List, Tuple, Optional
from collections import deque

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
