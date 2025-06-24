from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime

from .pose_detection import PoseDetection

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
