#!/usr/bin/env python3
"""
Fall Detection Evaluation Script - Generate Confusion Matrix
Evaluates fall detection accuracy across multiple video files
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse
from typing import List, Dict, Tuple
import cv2
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ai.mvp import HospitalMonitorMVP
from src.api.alert_service import alert_service
from video_labels_config import VIDEO_LABELS, EXCLUDE_VIDEOS

class FallDetectionEvaluator:
    """Evaluates fall detection performance and generates confusion matrix"""
    
    def __init__(self):
        self.results = []
        self.telegram_token = None
        self.chat_id = None
        
    def setup_telegram(self, token: str = None, chat_id: str = None):
        """Setup telegram credentials (optional for evaluation)"""
        self.telegram_token = token
        self.chat_id = chat_id
    
    def evaluate_single_video(self, video_path: str, true_label: str, room_id: str = None) -> Dict:
        """
        Evaluate a single video file
        
        Args:
            video_path: Path to video file
            true_label: Ground truth label ('fall' or 'no_fall')
            room_id: Room identifier for the video
            
        Returns:
            Dict with evaluation results
        """
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
            
        print(f"\n🎬 Evaluating: {video_path}")
        print(f"📋 True Label: {true_label}")
        
        # Clear previous alerts
        alert_service.alert_history.clear()
        
        # Use filename as room_id if not provided
        if room_id is None:
            room_id = Path(video_path).stem
        
        # Create monitor instance
        monitor = HospitalMonitorMVP(
            room_id=room_id,
            video_path=video_path,
            telegram_token=self.telegram_token,
            chat_id=self.chat_id
        )
        
        # Set very sensitive thresholds for testing
        monitor.inactivity_threshold = 0.5
        monitor.debug_mode = False  # Reduce noise during evaluation
        
        try:
            # Process video without display
            monitor.process_video_headless()
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return None
        
        # Analyze results - ONLY check for FALL_DETECTED alerts
        fall_alerts = [alert for alert in alert_service.alert_history 
                      if alert.alert_type == "FALL_DETECTED"]
        inactivity_alerts = [alert for alert in alert_service.alert_history 
                           if alert.alert_type == "PROLONGED_INACTIVITY"]
        
        # Determine predicted label based ONLY on fall detection alerts
        has_fall_alert = len(fall_alerts) > 0
        predicted_label = 'fall' if has_fall_alert else 'no_fall'
        
        # Calculate metrics
        is_correct = (predicted_label == true_label)
        
        result = {
            'video_path': video_path,
            'video_name': Path(video_path).name,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'fall_alerts_count': len(fall_alerts),
            'inactivity_alerts_count': len(inactivity_alerts),
            'total_alerts': len(alert_service.alert_history),
            'frames_processed': monitor.frame_count,
            'room_id': room_id,
            'has_fall_detection': has_fall_alert
        }
        
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"{status} | Fall Alert: {has_fall_alert} | Prediction: {predicted_label}")
        print(f"📊 Fall alerts: {len(fall_alerts)}, Inactivity alerts: {len(inactivity_alerts)}")
        
        return result
    
    def evaluate_dataset(self, video_configs: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple videos
        
        Args:
            video_configs: List of dicts with 'path' and 'label' keys
            
        Returns:
            DataFrame with results
        """
        print("🔍 Starting Fall Detection Evaluation")
        print("=" * 50)
        
        for config in video_configs:
            result = self.evaluate_single_video(
                video_path=config['path'],
                true_label=config['label'],
                room_id=config.get('room_id')
            )
            
            if result:
                self.results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        return df
    
    def generate_confusion_matrix(self, df: pd.DataFrame, save_path: str = None):
        """Generate and display confusion matrix"""
        if df.empty:
            print("❌ No results to generate confusion matrix")
            return
        
        # Get true and predicted labels
        y_true = df['true_label'].values
        y_pred = df['predicted_label'].values
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['fall', 'no_fall'])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, labels=['fall', 'no_fall'])
        
        # Print results
        print("\n📊 EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Total Videos Tested: {len(df)}")
        print(f"Correct Predictions: {df['is_correct'].sum()}")
        print(f"Incorrect Predictions: {(~df['is_correct']).sum()}")
        
        print("\n📈 Classification Report:")
        print(report)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fall', 'No Fall'], 
                   yticklabels=['Fall', 'No Fall'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Accuracy by video
        plt.subplot(2, 2, 2)
        correct_counts = df.groupby('true_label')['is_correct'].agg(['sum', 'count'])
        correct_counts['accuracy'] = correct_counts['sum'] / correct_counts['count']
        correct_counts['accuracy'].plot(kind='bar', color=['red', 'green'])
        plt.title('Accuracy by True Label')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Alert distribution
        plt.subplot(2, 2, 3)
        df.groupby('predicted_label')['fall_alerts_count'].mean().plot(kind='bar', color=['orange', 'blue'])
        plt.title('Average Fall Alerts by Prediction')
        plt.ylabel('Avg Fall Alerts')
        plt.xticks(rotation=45)
        
        # Detailed results table
        plt.subplot(2, 2, 4)
        plt.axis('tight')
        plt.axis('off')
        
        # Create summary table
        summary_data = []
        for label in ['fall', 'no_fall']:
            subset = df[df['true_label'] == label]
            summary_data.append([
                label,
                len(subset),
                subset['is_correct'].sum(),
                f"{subset['is_correct'].mean():.1%}",
                subset['fall_alerts_count'].mean()
            ])
        
        table = plt.table(cellText=summary_data,
                         colLabels=['True Label', 'Count', 'Correct', 'Accuracy', 'Avg Fall Alerts'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
        
        return cm, accuracy, report
    
    def save_detailed_results(self, df: pd.DataFrame, save_path: str = None):
        """Save detailed results to CSV"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"fall_detection_evaluation_{timestamp}.csv"
        
        df.to_csv(save_path, index=False)
        print(f"💾 Detailed results saved to: {save_path}")
        
        # Print detailed results
        print("\n📝 DETAILED RESULTS")
        print("=" * 80)
        for _, row in df.iterrows():
            status = "✅ CORRECT" if row['is_correct'] else "❌ INCORRECT"
            print(f"{status} | {row['video_name']} | True: {row['true_label']} | Pred: {row['predicted_label']} | Fall Alerts: {row['fall_alerts_count']}")

def create_video_dataset_config():
    """Create configuration for video dataset with labels from config file"""
    
    video_configs = []
    
    for video_path, label in VIDEO_LABELS.items():
        # Skip excluded videos
        if video_path in EXCLUDE_VIDEOS:
            continue
            
        # Create room_id from video path
        room_id = Path(video_path).stem.replace(' ', '_').replace('(', '').replace(')', '')
        
        video_configs.append({
            'path': video_path,
            'label': label,
            'room_id': room_id
        })
    
    return video_configs

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Fall Detection System')
    parser.add_argument('--telegram-token', help='Telegram bot token (optional)')
    parser.add_argument('--chat-id', help='Telegram chat ID (optional)')
    parser.add_argument('--save-plot', default='confusion_matrix.png', help='Save plot path')
    parser.add_argument('--save-csv', help='Save CSV results path')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FallDetectionEvaluator()
    evaluator.setup_telegram(args.telegram_token, args.chat_id)
    
    # Load video configurations
    video_configs = create_video_dataset_config()
    
    print(f"📹 Found {len(video_configs)} videos to evaluate")
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(video_configs)
    
    if not results_df.empty:
        # Generate confusion matrix
        cm, accuracy, report = evaluator.generate_confusion_matrix(results_df, args.save_plot)
        
        # Save detailed results
        evaluator.save_detailed_results(results_df, args.save_csv)
        
        # Print summary
        print(f"\n🎯 FINAL SUMMARY")
        print(f"Overall Accuracy: {accuracy:.1%}")
        print(f"Videos Tested: {len(results_df)}")
        
        # Identify problematic videos
        incorrect_videos = results_df[~results_df['is_correct']]
        if not incorrect_videos.empty:
            print(f"\n🚨 INCORRECT PREDICTIONS ({len(incorrect_videos)} videos):")
            for _, row in incorrect_videos.iterrows():
                print(f"  - {row['video_name']}: True={row['true_label']}, Predicted={row['predicted_label']}")
    else:
        print("❌ No videos were successfully processed")

if __name__ == "__main__":
    main()
