"""
AI module for hospital monitoring system.

This module contains the core AI components for patient monitoring:
- PoseDetection: Data structure for pose detection results
- PoseAnalyzer: Analyzes patient poses and detects anomalies
- SimpleTracker: Tracks patients across video frames
"""

from .pose_detection import PoseDetection
from .pose_analyzer import PoseAnalyzer
from .simple_tracker import SimpleTracker

__all__ = [
    'PoseDetection',
    'PoseAnalyzer', 
    'SimpleTracker'
]