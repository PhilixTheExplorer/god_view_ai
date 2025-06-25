import numpy as np
import math
from typing import List, Tuple, Optional
from collections import deque

class PoseAnalyzer:
    """Analyzes patient poses using keypoints and detects anomalies"""
    
    def __init__(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.floor_threshold = 0.80
        self.standing_angle_thresh = 30.0  # Angle threshold: <= 30° = lying (horizontal)
        self.lying_angle_thresh = 60.0     # Angle threshold: > 60° = standing (vertical)
        self.min_keypoints_needed = 5      # Min keypoints required for valid detection
        
        # COCO indexes
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
    
    def analyze_pose(self, bbox: Tuple[int, int, int, int], keypoints: np.ndarray) -> Tuple[str, float, float]:
        """Analyze posture and floor proximity"""
        x1, y1, x2, y2 = bbox
        floor_proximity = y2 / self.frame_height
        valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
        pose_confidence = np.mean(valid_keypoints[:, 2]) if len(valid_keypoints) > 0 else 0.0
        posture = self._classify_posture(keypoints, bbox)
        return posture, floor_proximity, pose_confidence

    def _classify_posture(self, keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Classify posture based on angles between shoulders, hips, and knees."""
        if len(keypoints) < self.min_keypoints_needed:
            return "unknown"

        head_y = self._get_keypoint_y(keypoints, ['nose', 'left_eye', 'right_eye'])
        shoulder_y = self._get_keypoint_y(keypoints, ['left_shoulder', 'right_shoulder'])
        hip_y = self._get_keypoint_y(keypoints, ['left_hip', 'right_hip'])
        knee_y = self._get_keypoint_y(keypoints, ['left_knee', 'right_knee'])
        ankle_y = self._get_keypoint_y(keypoints, ['left_ankle', 'right_ankle'])
        angle = self._calculate_body_angle(keypoints)

        if angle is None:
            return "unknown"

        # Fixed logic: smaller angles (close to horizontal) = lying, larger angles (close to vertical) = standing
        if angle <= self.standing_angle_thresh:  # Close to horizontal
            return "lying"
        elif angle > self.lying_angle_thresh:    # Close to vertical
            return "standing"
        else:
            return "sitting"

    def detect_fall(self, track_history: deque) -> bool:
        """Detect falls based on angle and floor proximity."""
        if len(track_history) < 3:
            return False
        recent = list(track_history)[-8:]

        angles = []
        floor_proximities = []
        for detection in recent:
            angles.append(self._calculate_body_angle(detection.keypoints))
            floor_proximities.append(detection.floor_proximity)

        lying_count = sum(1 for d in recent if d.posture == "lying")
        standing_count = sum(1 for d in recent if d.posture == "standing" or d.posture == "sitting")

        if lying_count >= 2 and standing_count >= 1 and max(floor_proximities) > 0.65:
            return True

        return False

    def detect_prolonged_inactivity(self, track_history: deque, threshold_seconds: int = 300) -> bool:
        """Detect prolonged inactivity based on position change over time, only for lying patients."""
        if len(track_history) < 10:
            return False

        recent = list(track_history)
        if not recent:
            return False

        # Check if patient has been lying for the majority of the time period
        lying_count = sum(1 for d in recent if d.posture == "lying")
        if lying_count < len(recent) * 0.7:  # Must be lying for at least 70% of the time
            return False

        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span.total_seconds() < threshold_seconds:
            return False

        first_center = self._get_bbox_center(recent[0].bbox)
        movement_threshold = 30
        for detection in recent[1:]:
            curr_center = self._get_bbox_center(detection.bbox)
            distance = math.sqrt((first_center[0] - curr_center[0])**2 + (first_center[1] - curr_center[1])**2)
            if distance > movement_threshold:
                return False
        return True

    def _get_keypoint_y(self, keypoints: np.ndarray, joint_names: List[str]) -> Optional[float]:
        """Get average Y coordinate of specified joints."""
        y_coords = []
        for name in joint_names:
            if name in self.KEYPOINTS:
                idx = self.KEYPOINTS[name]
                if idx < len(keypoints) and keypoints[idx, 2] > 0.3:
                    y_coords.append(keypoints[idx, 1])
        return np.mean(y_coords) if y_coords else None

    def _calculate_body_angle(self, keypoints: np.ndarray) -> Optional[float]:
        """Calculate body angle (degrees) from shoulder-to-hip line relative to horizontal.
        0° = completely horizontal (lying), 90° = completely vertical (standing)"""
        if (5 in range(len(keypoints)) and 6 in range(len(keypoints))
                and 11 in range(len(keypoints)) and 12 in range(len(keypoints))):

            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_hip = keypoints[11][:2]
            right_hip = keypoints[12][:2]

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            dy = hip_center[1] - shoulder_center[1]
            dx = hip_center[0] - shoulder_center[0]

            angle = math.degrees(math.atan2(dy, dx))
            return abs(angle)  # Return absolute angle relative to horizontal (0°=horizontal, 90°=vertical)
        return None

    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of the bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
