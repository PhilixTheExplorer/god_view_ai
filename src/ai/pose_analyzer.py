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
        """Enhanced fall detection with improved accuracy and reduced false positives."""
        if len(track_history) < 6:  # Need more frames for reliable detection
            return False
        
        recent = list(track_history)[-12:]  # Look at more frames for better context
        
        # Analyze posture transitions and movement patterns
        posture_sequence = [d.posture for d in recent]
        bbox_sequence = [d.bbox for d in recent]
        floor_proximities = [d.floor_proximity for d in recent]
        confidences = [d.confidence for d in recent]
        timestamps = [d.timestamp for d in recent]
        
        # Skip if too many unknown postures (unreliable detection)
        unknown_count = posture_sequence.count("unknown")
        if unknown_count > len(posture_sequence) * 0.4:  # More than 40% unknown
            return False
        
        # Check for minimum pose confidence
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < 0.4:  # Require decent pose detection quality
            return False
        
        # CRITICAL: Check for normal patterns first (these are NOT falls)
        if self._is_normal_movement_pattern(posture_sequence):
            print(f"[FALL DEBUG] Normal movement pattern detected - NOT a fall")
            return False
        
        # Enhanced fall detection with stricter criteria
        fall_score = 0.0
        
        # 1. PRIMARY INDICATOR: Direct standing to lying transition
        direct_fall = self._detect_direct_standing_to_lying(posture_sequence)
        if direct_fall:
            fall_score += 3.0  # Strong indicator
            print(f"[FALL DEBUG] Direct standing→lying transition: +3.0")
        
        # 1b. ALTERNATIVE PRIMARY: Rapid posture degradation (standing → sitting → lying quickly)
        rapid_degradation = self._detect_rapid_posture_degradation(posture_sequence, timestamps)
        if rapid_degradation:
            fall_score += 3.0  # Strong indicator
            print(f"[FALL DEBUG] Rapid posture degradation: +3.0")
        
        # 2. SECONDARY: Rapid vertical movement
        if (direct_fall or rapid_degradation) and len(bbox_sequence) >= 4:
            rapid_vertical = self._detect_rapid_vertical_movement(bbox_sequence)
            if rapid_vertical:
                fall_score += 1.0
                print(f"[FALL DEBUG] Rapid vertical movement: +1.0")
        
        # 3. TERTIARY: Aspect ratio change (body orientation)
        if len(bbox_sequence) >= 4:
            aspect_change = self._detect_body_orientation_change(bbox_sequence)
            if aspect_change:
                fall_score += 0.5
                print(f"[FALL DEBUG] Body orientation change: +0.5")
        
        # 4. VALIDATION: Temporal consistency check
        time_window = (timestamps[-1] - timestamps[0]).total_seconds()
        # More flexible time window for rapid degradation cases
        max_time_window = 5.0 if rapid_degradation else 3.0
        if time_window > max_time_window:
            print(f"[FALL DEBUG] Time window too long ({time_window:.1f}s > {max_time_window:.1f}s) - likely not a fall")
            return False
        
        # 5. VALIDATION: Final position check
        final_posture = posture_sequence[-1]
        if final_posture != "lying":
            print(f"[FALL DEBUG] Final posture is not lying ({final_posture}) - not a fall")
            return False
        
        print(f"[FALL DEBUG] Fall score: {fall_score:.1f}")
        print(f"[FALL DEBUG] Posture sequence: {posture_sequence}")
        print(f"[FALL DEBUG] Time window: {time_window:.1f}s")
        
        # Require high confidence for fall detection
        is_fall = fall_score >= 3.0
        
        if is_fall:
            print(f"[FALL DETECTION] ⚠️ FALL DETECTED! Score: {fall_score:.1f}")
            print(f"[FALL DETECTION] Sequence: {posture_sequence}")
            print(f"[FALL DETECTION] Duration: {time_window:.1f}s")
        
        return is_fall
    
    def _is_normal_movement_pattern(self, posture_sequence: List[str]) -> bool:
        """Check if the posture sequence represents normal, controlled movement (not a fall)."""
        if len(posture_sequence) < 3:
            return False
        
        # Pattern 1: Any lying to sitting/standing transition (getting up)
        for i in range(len(posture_sequence) - 1):
            if (posture_sequence[i] == "lying" and 
                posture_sequence[i + 1] in ["sitting", "standing"]):
                print(f"[FALL DEBUG] Normal getting up pattern: lying → {posture_sequence[i + 1]}")
                return True
        
        # Pattern 2: Lying → sitting → standing (getting up gradually)
        for i in range(len(posture_sequence) - 2):
            if (posture_sequence[i] == "lying" and 
                posture_sequence[i + 1] == "sitting" and 
                posture_sequence[i + 2] == "standing"):
                print(f"[FALL DEBUG] Normal getting up: lying → sitting → standing")
                return True
        
        # Pattern 3: Stable position (already lying for most of the sequence)
        lying_count = posture_sequence.count("lying")
        if lying_count >= len(posture_sequence) * 0.75:  # 75% lying
            print(f"[FALL DEBUG] Stable lying position (not a fall)")
            return True
        
        # REMOVED: Normal bedtime sequences - these can also be falls
        # The standing → sitting → lying pattern can occur during falls
        # We should rely on other fall detection indicators instead
        
        return False
    
    def _detect_direct_standing_to_lying(self, posture_sequence: List[str]) -> bool:
        """Detect direct standing to lying transition (primary fall indicator)."""
        # Look for standing directly followed by lying (skipping sitting)
        for i in range(len(posture_sequence) - 1):
            if (posture_sequence[i] == "standing" and 
                posture_sequence[i + 1] == "lying"):
                print(f"[FALL DEBUG] Direct standing→lying at position {i}")
                return True
        
        # Also check for standing → unknown → lying (detection gap during fall)
        for i in range(len(posture_sequence) - 2):
            if (posture_sequence[i] == "standing" and 
                posture_sequence[i + 1] == "unknown" and 
                posture_sequence[i + 2] == "lying"):
                print(f"[FALL DEBUG] Standing→unknown→lying at position {i}")
                return True
        
        # Check for standing → unknown → unknown → lying (longer detection gap)
        for i in range(len(posture_sequence) - 3):
            if (posture_sequence[i] == "standing" and 
                posture_sequence[i + 1] == "unknown" and 
                posture_sequence[i + 2] == "unknown" and 
                posture_sequence[i + 3] == "lying"):
                print(f"[FALL DEBUG] Standing→unknown→unknown→lying at position {i}")
                return True
        
        # Check for pattern where person was standing and later ends in lying with unknowns in between
        # This handles cases where fall detection is temporarily lost during the fall
        standing_positions = [i for i, p in enumerate(posture_sequence) if p == "standing"]
        lying_positions = [i for i, p in enumerate(posture_sequence) if p == "lying"]
        
        if standing_positions and lying_positions:
            last_standing = max(standing_positions)
            first_lying = min(lying_positions)
            
            # If standing was followed by lying with only unknowns in between (no sitting)
            if last_standing < first_lying:
                between_postures = posture_sequence[last_standing + 1:first_lying]
                if all(p in ["unknown"] for p in between_postures) and len(between_postures) <= 3:
                    print(f"[FALL DEBUG] Standing followed by lying with unknown gap: positions {last_standing}→{first_lying}")
                    return True
        
        return False
    
    def _detect_rapid_vertical_movement(self, bbox_sequence: List[Tuple[int, int, int, int]]) -> bool:
        """Detect rapid downward movement of the person's center."""
        if len(bbox_sequence) < 3:
            return False
        
        # Calculate center points and vertical movements
        centers = []
        for bbox in bbox_sequence:
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append(center_y)
        
        # Look for significant downward movement
        max_downward_movement = 0
        for i in range(1, len(centers)):
            movement = centers[i] - centers[i-1]
            if movement > max_downward_movement:
                max_downward_movement = movement
        
        # Threshold: movement should be at least 8% of frame height
        threshold = self.frame_height * 0.08
        
        if max_downward_movement > threshold:
            print(f"[FALL DEBUG] Rapid vertical movement: {max_downward_movement:.1f}px (threshold: {threshold:.1f}px)")
            return True
        
        return False
    
    def _detect_body_orientation_change(self, bbox_sequence: List[Tuple[int, int, int, int]]) -> bool:
        """Detect change in body orientation (vertical to horizontal)."""
        if len(bbox_sequence) < 3:
            return False
        
        # Calculate aspect ratios
        aspect_ratios = []
        for bbox in bbox_sequence:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if height > 0:
                ratio = width / height
                aspect_ratios.append(ratio)
        
        if len(aspect_ratios) < 3:
            return False
        
        # Look for transition from more vertical to more horizontal
        initial_ratio = aspect_ratios[0]
        final_ratio = aspect_ratios[-1]
        
        # Change from vertical/square (ratio < 1.2) to horizontal (ratio > 1.5)
        if initial_ratio < 1.2 and final_ratio > 1.5:
            print(f"[FALL DEBUG] Body orientation change: {initial_ratio:.2f} → {final_ratio:.2f}")
            return True
        
        return False

    def _detect_sudden_fall_transition(self, posture_sequence: List[str]) -> bool:
        """Legacy method - kept for compatibility. Use _detect_direct_standing_to_lying instead."""
        # This method is now simplified and mainly calls the new logic
        return self._detect_direct_standing_to_lying(posture_sequence)
    
    def _detect_rapid_horizontal_change(self, aspect_ratios: List[float]) -> bool:
        """Detect rapid change to horizontal orientation indicating a fall with improved sensitivity."""
        if len(aspect_ratios) < 4:
            return False
        
        print(f"[FALL DEBUG] Aspect ratios: {[f'{r:.2f}' for r in aspect_ratios]}")
        
        # Look for rapid transitions from vertical/square to horizontal
        for i in range(len(aspect_ratios) - 2):
            initial_ratio = aspect_ratios[i]
            mid_ratio = aspect_ratios[i + 1]
            final_ratio = aspect_ratios[i + 2]
            
            # Pattern 1: Sudden horizontal change
            if (initial_ratio < 1.3 and  # Was relatively vertical/square (more lenient)
                mid_ratio > 1.6 and     # Became horizontal (more sensitive)
                final_ratio > 1.6):     # Stayed horizontal
                print(f"[FALL DEBUG] Rapid horizontal change: {initial_ratio:.2f} → {mid_ratio:.2f} → {final_ratio:.2f}")
                return True
            
            # Pattern 2: Progressive horizontal change (gradual fall)
            if (initial_ratio < 1.2 and  # Started vertical
                mid_ratio > 1.4 and     # Transitioning
                final_ratio > 1.8):     # Ended very horizontal
                print(f"[FALL DEBUG] Progressive horizontal change: {initial_ratio:.2f} → {mid_ratio:.2f} → {final_ratio:.2f}")
                return True
        
        # Look for overall trend toward horizontal orientation
        if len(aspect_ratios) >= 5:
            early_avg = sum(aspect_ratios[:2]) / 2
            late_avg = sum(aspect_ratios[-2:]) / 2
            
            # Significant increase in horizontal orientation
            if early_avg < 1.2 and late_avg > 1.7:
                print(f"[FALL DEBUG] Overall horizontal trend: {early_avg:.2f} → {late_avg:.2f}")
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

    def _detect_rapid_posture_degradation(self, posture_sequence: List[str], timestamps: List) -> bool:
        """Detect rapid posture degradation (standing → sitting → lying) that indicates a fall."""
        if len(posture_sequence) < 3 or len(timestamps) < 3:
            return False
        
        # Look for standing → sitting → lying pattern
        for i in range(len(posture_sequence) - 2):
            if (posture_sequence[i] == "standing" and 
                posture_sequence[i + 1] == "sitting" and 
                posture_sequence[i + 2] == "lying"):
                
                # Check if this happened quickly (indicating a fall vs. normal lying down)
                time_span = (timestamps[i + 2] - timestamps[i]).total_seconds()
                
                # If transition happened within 2 seconds, it's likely a fall
                if time_span <= 2.0:
                    print(f"[FALL DEBUG] Rapid posture degradation: standing→sitting→lying in {time_span:.1f}s")
                    return True
                else:
                    print(f"[FALL DEBUG] Slow posture transition: standing→sitting→lying in {time_span:.1f}s (normal)")
        
        # Also check for standing → lying through sitting with some unknowns
        for i in range(len(posture_sequence) - 1):
            if posture_sequence[i] == "standing":
                # Look for lying position within next few frames
                for j in range(i + 1, min(i + 6, len(posture_sequence))):
                    if posture_sequence[j] == "lying":
                        # Check what's in between
                        between = posture_sequence[i + 1:j]
                        # If it contains sitting and/or unknowns (but no standing), and it's fast
                        if ("standing" not in between and 
                            any(p in ["sitting", "unknown"] for p in between)):
                            time_span = (timestamps[j] - timestamps[i]).total_seconds()
                            if time_span <= 2.5:
                                print(f"[FALL DEBUG] Rapid standing→lying with intermediate postures in {time_span:.1f}s")
                                return True
        
        return False
