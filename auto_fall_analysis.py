#!/usr/bin/env python3
"""
Automatic Fall Detection Evaluation
This script runs your fall detection system on all videos and automatically
determines ground truth based on whether FALL_DETECTED alerts are generated.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ai.mvp import HospitalMonitorMVP
from src.api.alert_service import alert_service

def get_all_video_files():
    """Get all available video files in the dataset"""
    video_files = []
    
    # Dataset chute02 videos
    chute02_path = Path("dataset/chute02")
    if chute02_path.exists():
        for video_file in chute02_path.glob("*.avi"):
            video_files.append(str(video_file))
    
    # Coffee room videos
    coffee_room_path = Path("FallDataset/Coffee_room_01/Videos")
    if coffee_room_path.exists():
        for video_file in coffee_room_path.glob("*.avi"):
            video_files.append(str(video_file))
    
    return sorted(video_files)

def process_single_video(video_path: str) -> Dict:
    """Process a single video and return results"""
    print(f"\n🎬 Processing: {video_path}")
    
    # Clear previous alerts
    alert_service.alert_history.clear()
    
    # Create monitor instance
    room_id = Path(video_path).stem.replace(' ', '_').replace('(', '').replace(')', '')
    monitor = HospitalMonitorMVP(
        room_id=room_id,
        video_path=video_path
    )
    
    # Set sensitive thresholds
    monitor.inactivity_threshold = 0.5
    monitor.debug_mode = False
    
    try:
        # Process video
        monitor.process_video_headless()
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None
    
    # Analyze results
    fall_alerts = [alert for alert in alert_service.alert_history 
                  if alert.alert_type == "FALL_DETECTED"]
    inactivity_alerts = [alert for alert in alert_service.alert_history 
                       if alert.alert_type == "PROLONGED_INACTIVITY"]
    
    has_fall_detection = len(fall_alerts) > 0
    label = 'fall' if has_fall_detection else 'no_fall'
    
    result = {
        'video_path': video_path,
        'video_name': Path(video_path).name,
        'detected_label': label,
        'fall_alerts_count': len(fall_alerts),
        'inactivity_alerts_count': len(inactivity_alerts),
        'total_alerts': len(alert_service.alert_history),
        'frames_processed': monitor.frame_count,
        'has_fall_detection': has_fall_detection
    }
    
    status = "🚨 FALL DETECTED" if has_fall_detection else "✅ NO FALL"
    print(f"{status} | Fall alerts: {len(fall_alerts)} | Total frames: {monitor.frame_count}")
    
    return result

def analyze_results(results: List[Dict]):
    """Analyze and display results"""
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No results to analyze")
        return
    
    # Summary statistics
    total_videos = len(df)
    fall_videos = len(df[df['detected_label'] == 'fall'])
    no_fall_videos = len(df[df['detected_label'] == 'no_fall'])
    
    print(f"\n📊 DETECTION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total Videos Processed: {total_videos}")
    print(f"Videos with Falls Detected: {fall_videos} ({fall_videos/total_videos:.1%})")
    print(f"Videos with No Falls: {no_fall_videos} ({no_fall_videos/total_videos:.1%})")
    
    # Fall detection details
    fall_df = df[df['detected_label'] == 'fall']
    if not fall_df.empty:
        print(f"\n🚨 VIDEOS WITH FALL DETECTION:")
        for _, row in fall_df.iterrows():
            print(f"  - {row['video_name']}: {row['fall_alerts_count']} fall alerts")
    
    # No fall details  
    no_fall_df = df[df['detected_label'] == 'no_fall']
    if not no_fall_df.empty:
        print(f"\n✅ VIDEOS WITH NO FALL DETECTION:")
        for _, row in no_fall_df.iterrows():
            print(f"  - {row['video_name']}: {row['inactivity_alerts_count']} inactivity alerts")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Detection summary pie chart
    plt.subplot(2, 2, 1)
    labels = ['Fall Detected', 'No Fall']
    sizes = [fall_videos, no_fall_videos]
    colors = ['red', 'green']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Fall Detection Results')
    
    # Fall alerts distribution
    plt.subplot(2, 2, 2)
    df['fall_alerts_count'].hist(bins=10, color='orange', alpha=0.7)
    plt.title('Distribution of Fall Alerts per Video')
    plt.xlabel('Number of Fall Alerts')
    plt.ylabel('Number of Videos')
    
    # Frames processed distribution
    plt.subplot(2, 2, 3)
    df['frames_processed'].hist(bins=10, color='blue', alpha=0.7)
    plt.title('Frames Processed per Video')
    plt.xlabel('Number of Frames')
    plt.ylabel('Number of Videos')
    
    # Summary table
    plt.subplot(2, 2, 4)
    plt.axis('tight')
    plt.axis('off')
    
    summary_data = [
        ['Total Videos', total_videos],
        ['Fall Detected', fall_videos],
        ['No Fall', no_fall_videos],
        ['Avg Fall Alerts', f"{df['fall_alerts_count'].mean():.1f}"],
        ['Avg Frames', f"{df['frames_processed'].mean():.0f}"]
    ]
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_file = f"fall_detection_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Results saved to: {csv_file}")
    
    # Save plot
    plot_file = f"fall_detection_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plot saved to: {plot_file}")
    
    plt.show()
    
    return df

def main():
    print("🎯 Automatic Fall Detection Analysis")
    print("=" * 50)
    
    # Get all video files
    video_files = get_all_video_files()
    
    if not video_files:
        print("❌ No video files found in dataset")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    for i, video in enumerate(video_files, 1):
        print(f"  {i:2d}. {video}")
    
    input("\n⏸️  Press Enter to start processing...")
    
    # Process all videos
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing...")
        result = process_single_video(video_path)
        if result:
            results.append(result)
    
    # Analyze results
    if results:
        print(f"\n🎯 ANALYSIS COMPLETE!")
        analyze_results(results)
    else:
        print("❌ No videos were successfully processed")

if __name__ == "__main__":
    main()
