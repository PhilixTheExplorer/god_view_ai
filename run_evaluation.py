#!/usr/bin/env python3
"""
Quick Fall Detection Evaluation Runner
Run this script to evaluate your fall detection system
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluate_fall_detection import FallDetectionEvaluator, create_video_dataset_config

def main():
    print("🎯 Fall Detection Evaluation")
    print("=" * 50)
    
    # Create evaluator
    evaluator = FallDetectionEvaluator()
    
    # Load video configurations
    video_configs = create_video_dataset_config()
    
    print(f"📹 Found {len(video_configs)} videos to evaluate")
    print("\n📋 Video List:")
    for i, config in enumerate(video_configs, 1):
        print(f"  {i:2d}. {config['path']} -> {config['label']}")
    
    input("\n⏸️  Press Enter to start evaluation...")
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(video_configs)
    
    if not results_df.empty:
        # Generate confusion matrix
        print("\n📊 Generating confusion matrix...")
        cm, accuracy, report = evaluator.generate_confusion_matrix(results_df, 'confusion_matrix.png')
        
        # Save detailed results
        evaluator.save_detailed_results(results_df, 'evaluation_results.csv')
        
        print(f"\n🎯 EVALUATION COMPLETE!")
        print(f"📈 Overall Accuracy: {accuracy:.1%}")
        print(f"📁 Results saved to: evaluation_results.csv")
        print(f"📊 Confusion matrix saved to: confusion_matrix.png")
        
        # Show problem videos
        incorrect_videos = results_df[~results_df['is_correct']]
        if not incorrect_videos.empty:
            print(f"\n🚨 VIDEOS WITH INCORRECT PREDICTIONS:")
            for _, row in incorrect_videos.iterrows():
                print(f"  ❌ {row['video_name']}: Expected {row['true_label']}, Got {row['predicted_label']}")
        else:
            print(f"\n✅ ALL PREDICTIONS CORRECT!")
            
    else:
        print("❌ No videos were successfully processed")

if __name__ == "__main__":
    main()
