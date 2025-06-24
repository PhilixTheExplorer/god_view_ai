import numpy as np
from datetime import datetime
from typing import Tuple
from dataclasses import dataclass

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
