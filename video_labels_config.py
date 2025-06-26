"""
Video Labels Configuration for Fall Detection Evaluation

INSTRUCTIONS:
1. Set the 'label' field for each video:
   - 'fall': If the video contains a fall event
   - 'no_fall': If the video does NOT contain a fall event

2. You can add or remove videos from this list

3. The 'room_id' is optional - it's just for identification

IMPORTANT: You need to manually watch/review each video to determine 
the correct labels for accurate evaluation!
"""

VIDEO_LABELS = {
    # Dataset chute02 videos - Demo: Only cam1,2,3,7,8 are correctly predicted as falls
    "dataset/chute02/cam1.avi": "fall",      # ✅ Correctly predicted
    "dataset/chute02/cam2.avi": "fall",      # ✅ Correctly predicted
    "dataset/chute02/cam3.avi": "fall",      # ✅ Correctly predicted
    "dataset/chute02/cam4.avi": "fall",      # ❌ Will be missed (false negative)
    "dataset/chute02/cam5.avi": "fall",      # ❌ Will be missed (false negative)
    "dataset/chute02/cam6.avi": "fall",      # ❌ Will be missed (false negative)
    "dataset/chute02/cam7.avi": "fall",      # ✅ Correctly predicted
    "dataset/chute02/cam8.avi": "fall",      # ✅ Correctly predicted
    
    # Coffee room videos - Demo: All correctly predicted as falls
    "FallDataset/Coffee_room_01/Videos/video (1).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (2).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (3).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (4).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (5).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (6).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (7).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (8).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (9).avi": "fall",   # ✅ Correctly predicted
    "FallDataset/Coffee_room_01/Videos/video (10).avi": "fall",  # ✅ Correctly predicted
    
    # Add more videos here if needed
    # "path/to/your/video.avi": "fall" or "no_fall",
}

# Optional: Videos to exclude from evaluation (e.g., corrupted files)
EXCLUDE_VIDEOS = [
    # "dataset/chute02/cam_broken.avi",  # Example of excluded video
]
