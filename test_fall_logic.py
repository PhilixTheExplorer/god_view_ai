#!/usr/bin/env python3
"""
Test script to validate the updated fall detection logic.
Tests the specific scenario mentioned in the user's debug output.
"""

import sys
import os
from datetime import datetime, timedelta
from collections import deque

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.pose_analyzer import PoseAnalyzer

class DetectionData:
    """Simple detection data class for testing."""
    def __init__(self, track_id, bbox, confidence, posture, keypoints, floor_proximity, timestamp):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.posture = posture
        self.keypoints = keypoints
        self.floor_proximity = floor_proximity
        self.timestamp = timestamp

def create_detection_data(posture, timestamp_offset=0, bbox=(100, 100, 200, 300)):
    """Create a DetectionData object for testing."""
    return DetectionData(
        track_id=15,
        bbox=bbox,
        confidence=0.8,
        posture=posture,
        keypoints=None,
        floor_proximity=0.8 if posture == "lying" else 0.2,
        timestamp=datetime.now() + timedelta(seconds=timestamp_offset)
    )

def test_fall_scenario():
    """Test the specific fall scenario from the user's debug output."""
    
    # Initialize the pose analyzer
    analyzer = PoseAnalyzer(frame_width=640, frame_height=480)
    
    print("="*60)
    print("TESTING FALL DETECTION - UPDATED LOGIC")
    print("="*60)
    
    # Test Case 1: Standing → Sitting → Lying sequence (should be detected as fall if rapid)
    print("\n1. Testing rapid standing → sitting → lying (SHOULD BE FALL):")
    print("-"*50)
    
    track_history = deque()
    
    # Add standing postures
    for i in range(4):
        track_history.append(create_detection_data("standing", i * 0.2))
    
    # Add sitting postures
    for i in range(2):
        track_history.append(create_detection_data("sitting", 4 * 0.2 + i * 0.2, bbox=(100, 150, 200, 300)))
    
    # Add lying postures
    for i in range(2):
        track_history.append(create_detection_data("lying", 6 * 0.2 + i * 0.2, bbox=(80, 200, 250, 280)))
    
    print(f"Posture sequence: {[d.posture for d in track_history]}")
    
    is_fall = analyzer.detect_fall(track_history)
    print(f"RESULT: {'✅ FALL DETECTED' if is_fall else '❌ NO FALL DETECTED'}")
    
    # Test Case 2: Same sequence but slower (should NOT be detected as fall)
    print("\n2. Testing slow standing → sitting → lying (SHOULD NOT BE FALL):")
    print("-"*50)
    
    track_history_slow = deque()
    
    # Add standing postures
    for i in range(4):
        track_history_slow.append(create_detection_data("standing", i * 1.0))  # 1 second intervals
    
    # Add sitting postures  
    for i in range(2):
        track_history_slow.append(create_detection_data("sitting", 4 + i * 1.0, bbox=(100, 150, 200, 300)))
    
    # Add lying postures
    for i in range(2):
        track_history_slow.append(create_detection_data("lying", 6 + i * 1.0, bbox=(80, 200, 250, 280)))
    
    print(f"Posture sequence: {[d.posture for d in track_history_slow]}")
    
    is_fall_slow = analyzer.detect_fall(track_history_slow)
    print(f"RESULT: {'✅ FALL DETECTED' if is_fall_slow else '❌ NO FALL DETECTED'}")
    
    # Test Case 3: Direct standing → lying (should definitely be detected)
    print("\n3. Testing direct standing → lying (SHOULD BE FALL):")
    print("-"*50)
    
    track_history_direct = deque()
    
    # Add standing postures
    for i in range(4):
        track_history_direct.append(create_detection_data("standing", i * 0.2))
    
    # Add lying postures directly
    for i in range(4):
        track_history_direct.append(create_detection_data("lying", 4 * 0.2 + i * 0.2, bbox=(80, 200, 250, 280)))
    
    print(f"Posture sequence: {[d.posture for d in track_history_direct]}")
    
    is_fall_direct = analyzer.detect_fall(track_history_direct)
    print(f"RESULT: {'✅ FALL DETECTED' if is_fall_direct else '❌ NO FALL DETECTED'}")
    
    # Test Case 4: Normal getting up (lying → sitting → standing - should NOT be detected)
    print("\n4. Testing normal getting up: lying → sitting → standing (SHOULD NOT BE FALL):")
    print("-"*50)
    
    track_history_getup = deque()
    
    # Add lying postures
    for i in range(4):
        track_history_getup.append(create_detection_data("lying", i * 0.5, bbox=(80, 200, 250, 280)))
    
    # Add sitting postures
    for i in range(2):
        track_history_getup.append(create_detection_data("sitting", 4 * 0.5 + i * 0.5, bbox=(100, 150, 200, 300)))
    
    # Add standing postures
    for i in range(2):
        track_history_getup.append(create_detection_data("standing", 6 * 0.5 + i * 0.5))
    
    print(f"Posture sequence: {[d.posture for d in track_history_getup]}")
    
    is_fall_getup = analyzer.detect_fall(track_history_getup)
    print(f"RESULT: {'✅ FALL DETECTED' if is_fall_getup else '❌ NO FALL DETECTED'}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"1. Rapid standing→sitting→lying: {'✅ DETECTED' if is_fall else '❌ MISSED'}")
    print(f"2. Slow standing→sitting→lying:  {'❌ FALSE POSITIVE' if is_fall_slow else '✅ CORRECTLY IGNORED'}")
    print(f"3. Direct standing→lying:        {'✅ DETECTED' if is_fall_direct else '❌ MISSED'}")
    print(f"4. Normal getting up:            {'❌ FALSE POSITIVE' if is_fall_getup else '✅ CORRECTLY IGNORED'}")
    
    # Determine overall success
    expected_results = [True, False, True, False]  # Expected outcomes for the 4 test cases
    actual_results = [is_fall, is_fall_slow, is_fall_direct, is_fall_getup]
    
    success_count = sum(1 for expected, actual in zip(expected_results, actual_results) if expected == actual)
    total_tests = len(expected_results)
    
    print(f"\nOVERALL: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Fall detection logic is working correctly.")
    else:
        print("⚠️  Some tests failed. Fall detection logic needs further adjustment.")
        
    return success_count == total_tests

if __name__ == "__main__":
    test_fall_scenario()
