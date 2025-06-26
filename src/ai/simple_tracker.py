from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
from .pose_detection import PoseDetection

class SimpleTracker:
    """Enhanced tracker for MVP - tracks people across frames with multi-angle CCTV support"""
    def __init__(self, max_age: int = 30, min_keypoints_needed: int = 5, max_disappeared: int = 10):
        self.tracks: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_age))
        self.last_seen: Dict[int, datetime] = {}
        self.disappeared_counts: Dict[int, int] = defaultdict(int)  # Track disappeared frames
        self.next_id = 1
        self.min_keypoints_needed = min_keypoints_needed
        self.max_disappeared = max_disappeared
        
        # Multi-angle tracking parameters
        self.position_weight = 0.4      # Weight for bounding box position similarity
        self.size_weight = 0.2          # Weight for bounding box size similarity
        self.keypoint_weight = 0.4      # Weight for keypoint similarity
        self.max_position_distance = 100  # Maximum pixel distance for position matching
        self.max_size_ratio = 2.0       # Maximum size change ratio

    def update(self, detections: List[PoseDetection]) -> List[PoseDetection]:
        """Update tracks with new detections, using enhanced multi-angle matching."""
        tracked_detections = []
        
        # Filter out detections with insufficient keypoints
        valid_detections = []
        for detection in detections:
            if (detection.keypoints.shape[0] >= self.min_keypoints_needed and 
                sum(detection.keypoints[:, 2] > 0.3) >= self.min_keypoints_needed):
                valid_detections.append(detection)
        
        # Update disappeared counts for existing tracks
        for track_id in list(self.tracks.keys()):
            self.disappeared_counts[track_id] += 1
        
        # Match detections to existing tracks
        used_detections = set()
        for detection in valid_detections:
            best_id = self._find_best_match_enhanced(detection)
            if best_id is not None:
                detection.id = best_id
                self.tracks[best_id].append(detection)
                self.last_seen[best_id] = detection.timestamp
                self.disappeared_counts[best_id] = 0  # Reset disappeared count
                tracked_detections.append(detection)
                used_detections.add(id(detection))
        
        # Create new tracks for unmatched detections
        for detection in valid_detections:
            if id(detection) not in used_detections:
                detection.id = self.next_id
                self.tracks[self.next_id].append(detection)
                self.last_seen[self.next_id] = detection.timestamp
                self.disappeared_counts[self.next_id] = 0
                tracked_detections.append(detection)
                self.next_id += 1
        
        # Remove tracks that have been missing for too long
        tracks_to_remove = []
        for track_id, disappeared_count in self.disappeared_counts.items():
            if disappeared_count > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.last_seen[track_id]
            del self.disappeared_counts[track_id]

        return tracked_detections

    def _find_best_match_enhanced(self, detection: PoseDetection) -> Optional[int]:
        """Find best matching track using enhanced multi-factor matching for CCTV views."""
        best_id = None
        best_score = 0.0
        min_score_threshold = 0.4  # Minimum score for a valid match
        
        det_bbox = detection.bbox
        det_keypoints = detection.keypoints
        
        for track_id, track_history in self.tracks.items():
            if not track_history or self.disappeared_counts[track_id] > 5:
                continue
                
            last_detection = track_history[-1]
            
            # Calculate multi-factor similarity score
            score = self._calculate_enhanced_similarity(det_bbox, det_keypoints, 
                                                      last_detection.bbox, last_detection.keypoints)
            
            if score > best_score and score > min_score_threshold:
                best_score = score
                best_id = track_id
                
        return best_id
    
    def _calculate_enhanced_similarity(self, bbox1: tuple, keypoints1: np.ndarray, 
                                     bbox2: tuple, keypoints2: np.ndarray) -> float:
        """Calculate enhanced similarity score using position, size, and keypoint features."""
        
        # 1. Position similarity (based on bounding box centers)
        center1 = self._get_bbox_center(bbox1)
        center2 = self._get_bbox_center(bbox2)
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Normalize distance score (closer = higher score)
        position_score = max(0, 1 - (distance / self.max_position_distance))
        
        # 2. Size similarity (based on bounding box areas)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        if area1 > 0 and area2 > 0:
            size_ratio = min(area1, area2) / max(area1, area2)
            size_score = size_ratio if size_ratio > (1 / self.max_size_ratio) else 0
        else:
            size_score = 0
        
        # 3. Keypoint similarity (based on key body parts)
        keypoint_score = self._calculate_keypoint_similarity(keypoints1, keypoints2)
        
        # 4. IoU similarity (original overlap measure)
        iou_score = self._calculate_overlap(bbox1, bbox2)
        
        # Combine scores with weights
        total_score = (self.position_weight * position_score + 
                      self.size_weight * size_score + 
                      self.keypoint_weight * keypoint_score + 
                      0.2 * iou_score)  # Small weight for IoU as backup
        
        return total_score
    
    def _calculate_keypoint_similarity(self, kp1: np.ndarray, kp2: np.ndarray) -> float:
        """Calculate similarity between keypoint patterns."""
        if len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        
        # Focus on stable keypoints: shoulders, hips, head
        key_indices = [0, 5, 6, 11, 12]  # nose, shoulders, hips
        
        similarities = []
        for idx in key_indices:
            if (idx < len(kp1) and idx < len(kp2) and 
                kp1[idx, 2] > 0.3 and kp2[idx, 2] > 0.3):
                
                # Calculate normalized distance between corresponding keypoints
                dist = ((kp1[idx, 0] - kp2[idx, 0])**2 + (kp1[idx, 1] - kp2[idx, 1])**2)**0.5
                # Normalize by image size assumption (can be improved with actual frame dimensions)
                normalized_dist = dist / 500.0  # Assume ~500px as reference
                similarity = max(0, 1 - normalized_dist)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _get_bbox_center(self, bbox: tuple) -> tuple:
        """Get center point of bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_overlap(self, bbox1: tuple, bbox2: tuple) -> float:
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
