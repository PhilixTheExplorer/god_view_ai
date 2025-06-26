import numpy as np
import math
from typing import List, Tuple, Optional
from collections import deque

class PoseAnalyzer:
    """Analyzes patient poses using keypoints and detects anomalies for CCTV camera views"""
    
    def __init__(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.floor_threshold = 0.80
        self.min_keypoints_needed = 5      # Min keypoints required for valid detection
        
        # COCO indexes
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Camera view thresholds for multi-angle detection
        self.lying_height_ratio_thresh = 0.4  # Height/width ratio threshold for lying detection
        self.bbox_aspect_ratio_lying = 2.0    # Width/height ratio for lying (wider than tall)
        self.vertical_distance_thresh = 0.3   # Normalized vertical distance threshold
        self.keypoint_spread_thresh = 0.15    # Keypoint spread threshold for pose classification
    
    def analyze_pose(self, bbox: Tuple[int, int, int, int], keypoints: np.ndarray) -> Tuple[str, float, float]:
        """Analyze posture and floor proximity"""
        x1, y1, x2, y2 = bbox
        floor_proximity = y2 / self.frame_height
        valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
        pose_confidence = np.mean(valid_keypoints[:, 2]) if len(valid_keypoints) > 0 else 0.0
        posture = self._classify_posture(keypoints, bbox)
        return posture, floor_proximity, pose_confidence

    def _classify_posture(self, keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Classify posture based on multi-angle CCTV camera view analysis."""
        if len(keypoints) < self.min_keypoints_needed:
            return "unknown"
        
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        
        # Detect camera viewing angle
        viewing_angle = self._estimate_viewing_angle(keypoints)
        
        # Get key body part positions
        head_pos = self._get_keypoint_avg(keypoints, ['nose', 'left_eye', 'right_eye'])
        shoulder_pos = self._get_keypoint_avg(keypoints, ['left_shoulder', 'right_shoulder'])
        hip_pos = self._get_keypoint_avg(keypoints, ['left_hip', 'right_hip'])
        knee_pos = self._get_keypoint_avg(keypoints, ['left_knee', 'right_knee'])
        ankle_pos = self._get_keypoint_avg(keypoints, ['left_ankle', 'right_ankle'])
        
        # Calculate relative positions and distances
        body_parts = [head_pos, shoulder_pos, hip_pos, knee_pos, ankle_pos]
        valid_parts = [part for part in body_parts if part is not None]
        
        if len(valid_parts) < 3:
            return "unknown"
        
        # Multi-angle posture classification
        if viewing_angle in ['front', 'back']:
            return self._classify_frontal_view(keypoints, bbox, valid_parts)
        elif viewing_angle in ['left_side', 'right_side']:
            return self._classify_side_view(keypoints, bbox, valid_parts)
        elif viewing_angle in ['diagonal_front', 'diagonal_back']:
            return self._classify_diagonal_view(keypoints, bbox, valid_parts)
        else:
            # Fallback to general classification
            return self._classify_general_view(keypoints, bbox, valid_parts)
    
    def _estimate_viewing_angle(self, keypoints: np.ndarray) -> str:
        """Estimate camera viewing angle based on keypoint visibility and positions."""
        # Check shoulder visibility and relative positions
        left_shoulder = keypoints[5] if 5 < len(keypoints) else None
        right_shoulder = keypoints[6] if 6 < len(keypoints) else None
        left_hip = keypoints[11] if 11 < len(keypoints) else None
        right_hip = keypoints[12] if 12 < len(keypoints) else None
        
        # Count visible keypoints on each side
        left_side_visible = sum(1 for idx in [1, 3, 5, 7, 9, 11, 13, 15] 
                              if idx < len(keypoints) and keypoints[idx, 2] > 0.3)
        right_side_visible = sum(1 for idx in [2, 4, 6, 8, 10, 12, 14, 16] 
                               if idx < len(keypoints) and keypoints[idx, 2] > 0.3)
        
        # Analyze shoulder width visibility
        if (left_shoulder is not None and right_shoulder is not None and 
            left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3):
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            # Wide shoulder separation indicates frontal/back view
            if shoulder_width > 40:  # Pixels threshold
                # Check if face is visible to distinguish front vs back
                nose_visible = 0 < len(keypoints) and keypoints[0, 2] > 0.3
                eyes_visible = any(keypoints[i, 2] > 0.3 for i in [1, 2] if i < len(keypoints))
                
                if nose_visible or eyes_visible:
                    return 'front'
                else:
                    return 'back'
            else:
                # Narrow shoulder separation indicates side view
                if left_side_visible > right_side_visible:
                    return 'left_side'
                elif right_side_visible > left_side_visible:
                    return 'right_side'
                else:
                    return 'diagonal_front'
        
        # Fallback based on keypoint visibility
        if abs(left_side_visible - right_side_visible) >= 2:
            if left_side_visible > right_side_visible:
                return 'left_side'
            else:
                return 'right_side'
        
        return 'front'  # Default assumption
    
    def _classify_frontal_view(self, keypoints: np.ndarray, bbox: tuple, valid_parts: list) -> str:
        """Classify posture for frontal/back camera view."""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        # For frontal view, use vertical alignment of key body parts
        head_y = self._get_keypoint_y(keypoints, ['nose', 'left_eye', 'right_eye'])
        shoulder_y = self._get_keypoint_y(keypoints, ['left_shoulder', 'right_shoulder'])
        hip_y = self._get_keypoint_y(keypoints, ['left_hip', 'right_hip'])
        ankle_y = self._get_keypoint_y(keypoints, ['left_ankle', 'right_ankle'])
        
        # Calculate vertical progression (head should be above shoulders, shoulders above hips, etc.)
        if head_y and shoulder_y and hip_y:
            head_shoulder_dist = abs(head_y - shoulder_y) / self.frame_height
            shoulder_hip_dist = abs(shoulder_y - hip_y) / self.frame_height
            
            # Check if person is upright (normal vertical alignment)
            if (head_y < shoulder_y < hip_y and  # Normal top-to-bottom order
                head_shoulder_dist > 0.05 and shoulder_hip_dist > 0.08):
                return "standing"
            
            # Check for sitting (compressed vertical alignment)
            elif (head_y < shoulder_y and shoulder_hip_dist < 0.15 and
                  bbox_height / self.frame_height < 0.7):
                return "sitting"
            
            # Check for lying (minimal vertical alignment or horizontal orientation)
            elif (head_shoulder_dist < 0.08 or bbox_width > bbox_height * 1.5):
                return "lying"
        
        # Fallback based on bounding box aspect ratio
        if bbox_width > bbox_height * self.bbox_aspect_ratio_lying:
            return "lying"
        elif bbox_height > bbox_width * 1.2:
            return "standing"
        else:
            return "sitting"
    
    def _classify_side_view(self, keypoints: np.ndarray, bbox: tuple, valid_parts: list) -> str:
        """Classify posture for side camera view."""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        # For side view, analyze the spine curve and body alignment
        shoulder_pos = self._get_keypoint_avg(keypoints, ['left_shoulder', 'right_shoulder'])
        hip_pos = self._get_keypoint_avg(keypoints, ['left_hip', 'right_hip'])
        ankle_pos = self._get_keypoint_avg(keypoints, ['left_ankle', 'right_ankle'])
        
        if shoulder_pos and hip_pos:
            # Calculate body angle from side view
            dx = hip_pos[0] - shoulder_pos[0]
            dy = hip_pos[1] - shoulder_pos[1]
            body_angle = math.degrees(math.atan2(abs(dy), abs(dx))) if dx != 0 else 90
            
            # Analyze vertical vs horizontal alignment
            if body_angle > 70:  # Nearly vertical alignment
                return "standing"
            elif body_angle < 30:  # Nearly horizontal alignment
                return "lying"
            else:
                # Check ankle position relative to hip for sitting detection
                if ankle_pos and ankle_pos[1] < hip_pos[1]:  # Ankles above hips suggests sitting
                    return "sitting"
                else:
                    return "standing"
        
        # Fallback based on aspect ratio
        if bbox_width > bbox_height * 1.8:
            return "lying"
        elif bbox_height > bbox_width * 1.3:
            return "standing"
        else:
            return "sitting"
    
    def _classify_diagonal_view(self, keypoints: np.ndarray, bbox: tuple, valid_parts: list) -> str:
        """Classify posture for diagonal camera view."""
        # Combine frontal and side view analysis techniques
        frontal_result = self._classify_frontal_view(keypoints, bbox, valid_parts)
        side_result = self._classify_side_view(keypoints, bbox, valid_parts)
        
        # Use agreement between methods, or default to more conservative estimate
        if frontal_result == side_result:
            return frontal_result
        elif "lying" in [frontal_result, side_result]:
            return "lying"  # Prioritize lying detection for safety
        elif "standing" in [frontal_result, side_result]:
            return "standing"
        else:
            return "sitting"
    
    def _classify_general_view(self, keypoints: np.ndarray, bbox: tuple, valid_parts: list) -> str:
        """General posture classification as fallback."""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        # Use bounding box aspect ratio as primary indicator
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        
        if aspect_ratio > 1.8:  # Very wide bounding box
            return "lying"
        elif aspect_ratio < 0.6:  # Very tall bounding box
            return "standing"
        else:
            # Analyze keypoint spread
            y_coords = [kp[1] for kp in valid_parts]
            y_spread = (max(y_coords) - min(y_coords)) / self.frame_height if y_coords else 0
            
            if y_spread > 0.4:  # Large vertical spread
                return "standing"
            elif y_spread < 0.2:  # Small vertical spread
                return "lying"
            else:
                return "sitting"

    def detect_fall(self, track_history: deque) -> bool:
        """Detect falls based on multi-angle analysis and movement patterns."""
        if len(track_history) < 3:
            return False
        recent = list(track_history)[-8:]

        # Analyze posture transitions and movement patterns
        posture_sequence = [d.posture for d in recent]
        bbox_sequence = [d.bbox for d in recent]
        floor_proximities = [d.floor_proximity for d in recent]
        
        # Debug logging
        print(f"[DEBUG] Fall detection analysis for sequence: {posture_sequence}")
        
        # Enhanced fall detection criteria
        fall_indicators = 0
        
        # 1. SUDDEN posture transition: standing/sitting to lying (not gradual)
        # Check for rapid, uncontrolled transitions vs normal movement
        sudden_transition = self._detect_sudden_fall_transition(posture_sequence)
        if sudden_transition:
            fall_indicators += 1
            print(f"[DEBUG] Sudden transition detected: +1 indicator")
        
        # 2. Rapid vertical movement (based on bounding box center changes)
        if len(bbox_sequence) >= 3:
            vertical_movements = []
            for i in range(1, len(bbox_sequence)):
                prev_center_y = (bbox_sequence[i-1][1] + bbox_sequence[i-1][3]) / 2
                curr_center_y = (bbox_sequence[i][1] + bbox_sequence[i][3]) / 2
                vertical_movements.append(curr_center_y - prev_center_y)
            
            # Check for significant downward movement (falling motion)
            max_downward = max(vertical_movements) if vertical_movements else 0
            print(f"[DEBUG] Max downward movement: {max_downward} (threshold: {self.frame_height * 0.1})")
            if max_downward > self.frame_height * 0.1:  # 10% of frame height
                fall_indicators += 1
                print(f"[DEBUG] Rapid vertical movement detected: +1 indicator")
        
        # 3. High floor proximity in lying position (after sudden movement)
        recent_lying = [d for d in recent if d.posture == "lying"]
        if recent_lying:
            max_floor_prox = max(d.floor_proximity for d in recent_lying)
            print(f"[DEBUG] Max floor proximity: {max_floor_prox} (threshold: 0.7)")
            if max_floor_prox > 0.7:
                fall_indicators += 1
                print(f"[DEBUG] High floor proximity detected: +1 indicator")
        
        # 4. Rapid bounding box aspect ratio change (sudden horizontal orientation)
        if len(bbox_sequence) >= 3:
            aspect_ratios = []
            for bbox in bbox_sequence:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratios.append(width / height if height > 0 else 1.0)
            
            print(f"[DEBUG] Aspect ratios: {[round(r, 2) for r in aspect_ratios]}")
            # Check for rapid increase in aspect ratio (sudden horizontal orientation)
            if self._detect_rapid_horizontal_change(aspect_ratios):
                fall_indicators += 1
                print(f"[DEBUG] Rapid horizontal change detected: +1 indicator")
        
        print(f"[DEBUG] Total fall indicators: {fall_indicators}/4")
        
        # Enhanced fall criteria: require at least 2 indicators for better accuracy
        # BUT for sitting → lying falls, be more sensitive since they're often real falls
        sitting_to_lying_fall = any(
            recent[i].posture == "sitting" and recent[i+1].posture == "lying" 
            for i in range(len(recent)-1)
        )
        
        # Lower threshold for sitting → lying falls (they're often real)
        required_indicators = 1 if sitting_to_lying_fall else 2
        result = fall_indicators >= required_indicators
        
        if result:
            print(f"[FALL DETECTION] Sequence: {posture_sequence} -> FALL DETECTED ({fall_indicators} indicators, threshold: {required_indicators})")
        else:
            print(f"[FALL DETECTION] Sequence: {posture_sequence} -> NO FALL ({fall_indicators} indicators, threshold: {required_indicators})")
        return result
    
    def _detect_sudden_fall_transition(self, posture_sequence: List[str]) -> bool:
        """Detect sudden fall transitions vs normal movement patterns."""
        if len(posture_sequence) < 3:
            return False
        
        # print(f"[DEBUG] Analyzing posture sequence: {posture_sequence}")
        
        # STEP 1: Look for fall patterns FIRST (standing/sitting → lying)
        fall_detected = False
        fall_position = -1
        
        for i in range(len(posture_sequence) - 1):
            # Pattern 1: Direct standing to lying (sudden collapse)
            if (posture_sequence[i] == "standing" and 
                posture_sequence[i + 1] == "lying"):
                print(f"[FALL DEBUG] Sudden collapse detected: standing → lying at position {i}")
                fall_detected = True
                fall_position = i + 1  # Position where lying starts
                break
            # Pattern 2: Sitting to lying (could also be a fall from sitting)
            elif (posture_sequence[i] == "sitting" and 
                  posture_sequence[i + 1] == "lying"):
                print(f"[FALL DEBUG] Potential sitting fall detected: sitting → lying at position {i}")
                fall_detected = True
                fall_position = i + 1  # Position where lying starts
                break
        
        # STEP 2: If no fall detected, check for normal movement patterns
        if not fall_detected:
            # Check for normal lying to sitting/standing transitions (getting up normally)
            for i in range(len(posture_sequence) - 1):
                if (posture_sequence[i] == "lying" and 
                    posture_sequence[i + 1] in ["sitting", "standing"]):
                    # Only consider this normal if it's not preceded by a recent fall
                    print(f"[FALL DEBUG] Normal movement detected (no prior fall): lying → {posture_sequence[i + 1]} at position {i}")
                    return False
            
            # Check for gradual transitions (controlled movement)
            for i in range(len(posture_sequence) - 2):
                # Pattern: lying → sitting → standing (normal getting up sequence)
                if (posture_sequence[i] == "lying" and 
                    posture_sequence[i + 1] == "sitting" and 
                    posture_sequence[i + 2] == "standing"):
                    print(f"[FALL DEBUG] Normal getting up sequence detected: lying → sitting → standing")
                    return False
        
        # STEP 3: If fall was detected, check if subsequent movements invalidate it
        if fall_detected:
            # Allow struggling movements after a fall (lying → sitting is normal after falling)
            # But don't allow complete recovery (lying → standing) immediately
            for i in range(fall_position, len(posture_sequence) - 1):
                if (posture_sequence[i] == "lying" and 
                    posture_sequence[i + 1] == "standing"):
                    # Direct lying to standing after a fall might indicate normal movement
                    # But only if it's more than 2 frames after the fall
                    if i - fall_position > 2:
                        print(f"[FALL DEBUG] Quick recovery detected: lying → standing at position {i} (too quick, might be normal movement)")
                        return False
            
            # Fall detected and no immediate full recovery - this is likely a real fall
            return True
        
        # STEP 4: Check for erratic movement patterns if no direct fall was found
        # Pattern: Multiple erratic movements ending in lying (loss of balance)
        if len(posture_sequence) >= 4:
            for i in range(len(posture_sequence) - 3):
                window = posture_sequence[i:i+4]
                unique_postures = set(window)
                
                # If we see 3+ different postures in 4 frames ending with lying
                # AND it started from standing/sitting (not already lying)
                if (len(unique_postures) >= 3 and 
                    window[-1] == "lying" and 
                    window[-2] == "lying" and
                    "standing" in window[:2]):  # Started from standing
                    
                    print(f"[FALL DEBUG] Erratic movement ending in lying detected: {window}")
                    return True
        
        # No fall patterns detected
        return False
    
    def _detect_rapid_horizontal_change(self, aspect_ratios: List[float]) -> bool:
        """Detect rapid change to horizontal orientation indicating a fall."""
        if len(aspect_ratios) < 3:
            return False
        
        # Check for sudden increase in aspect ratio (becoming more horizontal)
        for i in range(len(aspect_ratios) - 2):
            # Rapid change from vertical/square to horizontal orientation
            if (aspect_ratios[i] < 1.2 and  # Was relatively vertical/square
                aspect_ratios[i + 1] > 1.8 and  # Became horizontal
                aspect_ratios[i + 2] > 1.8):  # Stayed horizontal
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
    
    def _get_keypoint_avg(self, keypoints: np.ndarray, joint_names: List[str]) -> Optional[Tuple[float, float]]:
        """Get average position (x, y) of specified joints."""
        positions = []
        for name in joint_names:
            if name in self.KEYPOINTS:
                idx = self.KEYPOINTS[name]
                if idx < len(keypoints) and keypoints[idx, 2] > 0.3:
                    positions.append(keypoints[idx, :2])
        
        if positions:
            avg_pos = np.mean(positions, axis=0)
            return (float(avg_pos[0]), float(avg_pos[1]))
        return None

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
