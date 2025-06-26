"""
FallDataset Ground Truth Loader for Confusion Matrix Analysis

This module loads ground truth annotations from the FallDataset format:
- Line 1: Start frame of fall
- Line 2: End frame of fall  
- Following lines: frame_number,person_id,x1,y1,x2,y2 (bounding box coordinates)

Format: 320x240 resolution, 25 FPS
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import cv2
import numpy as np


class FallDatasetGroundTruth:
    """
    Loads and parses ground truth annotations from FallDataset format
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the ground truth loader
        
        Args:
            dataset_path: Path to the dataset folder (e.g., "FallDataset/Coffee_room_01")
        """
        self.dataset_path = Path(dataset_path)
        self.videos_path = self.dataset_path / "Videos"
        self.annotations_path = self.dataset_path / "Annotation_files"
        
        # Video properties from FallDataset specification
        self.frame_width = 320
        self.frame_height = 240
        self.fps = 25
        
        # Cache for loaded annotations
        self._annotation_cache = {}
    
    def get_available_videos(self) -> List[str]:
        """
        Get list of available video files
        
        Returns:
            List of video filenames
        """
        if not self.videos_path.exists():
            return []
        
        videos = []
        for video_file in self.videos_path.glob("*.avi"):
            annotation_file = self.annotations_path / f"{video_file.stem}.txt"
            if annotation_file.exists():
                videos.append(video_file.name)
        
        return sorted(videos)
    
    def load_annotation(self, video_name: str) -> Dict:
        """
        Load annotation for a specific video
        
        Args:
            video_name: Name of the video file (e.g., "video (1).avi")
            
        Returns:
            Dictionary containing:
            - fall_start_frame: Frame where fall begins
            - fall_end_frame: Frame where fall ends
            - bounding_boxes: Dict[frame_number] = (x1, y1, x2, y2)
            - fall_frames: Set of frame numbers where fall occurs
        """
        if video_name in self._annotation_cache:
            return self._annotation_cache[video_name]
        
        # Get annotation file path
        video_stem = Path(video_name).stem
        annotation_file = self.annotations_path / f"{video_stem}.txt"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        annotation_data = {
            'fall_start_frame': None,
            'fall_end_frame': None,
            'bounding_boxes': {},
            'fall_frames': set(),
            'total_frames': 0
        }
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        # Parse fall start and end frames
        if len(lines) >= 2:
            annotation_data['fall_start_frame'] = int(lines[0].strip())
            annotation_data['fall_end_frame'] = int(lines[1].strip())
            
            # Create set of fall frames
            annotation_data['fall_frames'] = set(range(
                annotation_data['fall_start_frame'],
                annotation_data['fall_end_frame'] + 1
            ))
        
        # Parse bounding box data
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) >= 6:
                frame_number = int(parts[0])
                person_id = int(parts[1])
                x1, y1, x2, y2 = map(int, parts[2:6])
                
                # Only store valid bounding boxes (not 0,0,0,0)
                if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                    annotation_data['bounding_boxes'][frame_number] = (x1, y1, x2, y2)
                
                annotation_data['total_frames'] = max(annotation_data['total_frames'], frame_number)
        
        # Cache the result
        self._annotation_cache[video_name] = annotation_data
        return annotation_data
    
    def get_ground_truth_for_frame(self, video_name: str, frame_number: int) -> Dict:
        """
        Get ground truth information for a specific frame
        
        Args:
            video_name: Name of the video file
            frame_number: Frame number (1-based)
            
        Returns:
            Dictionary containing:
            - is_fall: Boolean indicating if fall occurs in this frame
            - bounding_box: Tuple (x1, y1, x2, y2) or None
            - posture: Estimated posture based on bounding box
        """
        annotation = self.load_annotation(video_name)
        
        result = {
            'is_fall': frame_number in annotation['fall_frames'],
            'bounding_box': annotation['bounding_boxes'].get(frame_number),
            'posture': 'unknown'
        }
        
        # Estimate posture from bounding box
        if result['bounding_box']:
            x1, y1, x2, y2 = result['bounding_box']
            width = x2 - x1
            height = y2 - y1
            
            if height > 0:
                aspect_ratio = width / height
                
                # Simple posture estimation based on bounding box aspect ratio
                if aspect_ratio > 1.5:  # Wide bounding box
                    result['posture'] = 'lying'
                elif aspect_ratio < 0.6:  # Tall bounding box
                    result['posture'] = 'standing'
                else:
                    result['posture'] = 'sitting'
        
        return result
    
    def get_video_info(self, video_name: str) -> Dict:
        """
        Get video information and statistics
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Dictionary with video information
        """
        annotation = self.load_annotation(video_name)
        video_path = self.videos_path / video_name
        
        # Get actual video properties
        cap = cv2.VideoCapture(str(video_path))
        actual_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            'video_path': str(video_path),
            'annotation_path': str(self.annotations_path / f"{Path(video_name).stem}.txt"),
            'fall_start_frame': annotation['fall_start_frame'],
            'fall_end_frame': annotation['fall_end_frame'],
            'fall_duration_frames': len(annotation['fall_frames']),
            'fall_duration_seconds': len(annotation['fall_frames']) / self.fps,
            'annotated_frames': len(annotation['bounding_boxes']),
            'total_frames_annotation': annotation['total_frames'],
            'total_frames_actual': actual_total_frames,
            'fps_expected': self.fps,
            'fps_actual': actual_fps,
            'resolution_expected': (self.frame_width, self.frame_height),
            'resolution_actual': (actual_width, actual_height)
        }
    
    def create_evaluation_dataset(self, video_names: Optional[List[str]] = None) -> Dict:
        """
        Create a comprehensive evaluation dataset
        
        Args:
            video_names: List of video names to include, or None for all available
            
        Returns:
            Dictionary with evaluation data for all specified videos
        """
        if video_names is None:
            video_names = self.get_available_videos()
        
        evaluation_data = {
            'videos': {},
            'summary': {
                'total_videos': len(video_names),
                'total_falls': 0,
                'total_frames': 0,
                'fall_frames': 0,
                'non_fall_frames': 0
            }
        }
        
        for video_name in video_names:
            try:
                annotation = self.load_annotation(video_name)
                video_info = self.get_video_info(video_name)
                
                evaluation_data['videos'][video_name] = {
                    'annotation': annotation,
                    'info': video_info
                }
                
                # Update summary
                evaluation_data['summary']['total_falls'] += 1 if annotation['fall_frames'] else 0
                evaluation_data['summary']['total_frames'] += video_info['total_frames_actual']
                evaluation_data['summary']['fall_frames'] += len(annotation['fall_frames'])
                evaluation_data['summary']['non_fall_frames'] += (
                    video_info['total_frames_actual'] - len(annotation['fall_frames'])
                )
                
            except Exception as e:
                print(f"Warning: Could not load {video_name}: {e}")
        
        return evaluation_data
    
    def print_dataset_summary(self, video_names: Optional[List[str]] = None):
        """
        Print a summary of the dataset
        
        Args:
            video_names: List of video names to analyze, or None for all
        """
        if video_names is None:
            video_names = self.get_available_videos()
        
        print("=" * 60)
        print("📊 FallDataset Summary")
        print("=" * 60)
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Total Videos: {len(video_names)}")
        print()
        
        total_falls = 0
        total_frames = 0
        total_fall_frames = 0
        
        for video_name in video_names[:5]:  # Show first 5 videos as examples
            try:
                annotation = self.load_annotation(video_name)
                video_info = self.get_video_info(video_name)
                
                print(f"📹 {video_name}:")
                print(f"   Fall: Frames {annotation['fall_start_frame']}-{annotation['fall_end_frame']}")
                print(f"   Total Frames: {video_info['total_frames_actual']}")
                print(f"   Fall Duration: {len(annotation['fall_frames'])} frames ({len(annotation['fall_frames'])/self.fps:.1f}s)")
                print(f"   Annotated Frames: {annotation['total_frames']}")
                print()
                
                total_falls += 1
                total_frames += video_info['total_frames_actual']
                total_fall_frames += len(annotation['fall_frames'])
                
            except Exception as e:
                print(f"   Error loading {video_name}: {e}")
        
        if len(video_names) > 5:
            print(f"... and {len(video_names) - 5} more videos")
            
            # Calculate totals for all videos
            evaluation_data = self.create_evaluation_dataset(video_names)
            total_falls = evaluation_data['summary']['total_falls']
            total_frames = evaluation_data['summary']['total_frames']
            total_fall_frames = evaluation_data['summary']['fall_frames']
        
        print("=" * 60)
        print("📈 Dataset Statistics:")
        print(f"   Total Falls: {total_falls}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Fall Frames: {total_fall_frames} ({total_fall_frames/total_frames*100:.1f}%)")
        print(f"   Non-Fall Frames: {total_frames - total_fall_frames} ({(total_frames - total_fall_frames)/total_frames*100:.1f}%)")
        print("=" * 60)


def demo_ground_truth_loader():
    """
    Demonstration of the ground truth loader
    """
    print("🔍 FallDataset Ground Truth Loader Demo")
    print("=" * 50)
    
    # Initialize loader
    gt_loader = FallDatasetGroundTruth("FallDataset/Coffee_room_01")
    
    # Print dataset summary
    gt_loader.print_dataset_summary()
    
    # Demo specific video analysis
    videos = gt_loader.get_available_videos()
    if videos:
        demo_video = videos[0]
        print(f"\n📹 Detailed Analysis of {demo_video}:")
        print("-" * 40)
        
        # Load annotation
        annotation = gt_loader.load_annotation(demo_video)
        print(f"Fall Period: Frames {annotation['fall_start_frame']} - {annotation['fall_end_frame']}")
        print(f"Total Fall Frames: {len(annotation['fall_frames'])}")
        print(f"Annotated Frames: {len(annotation['bounding_boxes'])}")
        
        # Show some frame examples
        print(f"\n📋 Sample Frame Analysis:")
        test_frames = [1, annotation['fall_start_frame'], annotation['fall_end_frame'], annotation['total_frames']]
        
        for frame_num in test_frames:
            if frame_num <= annotation['total_frames']:
                gt_data = gt_loader.get_ground_truth_for_frame(demo_video, frame_num)
                print(f"   Frame {frame_num}: Fall={gt_data['is_fall']}, Posture={gt_data['posture']}")
                if gt_data['bounding_box']:
                    x1, y1, x2, y2 = gt_data['bounding_box']
                    print(f"                 BBox=({x1},{y1},{x2},{y2})")
    
    return gt_loader


if __name__ == "__main__":
    demo_ground_truth_loader()
