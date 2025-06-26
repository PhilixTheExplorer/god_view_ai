#!/usr/bin/env python3
"""
Demo Confusion Matrix Generator
Creates confusion matrix based on predefined demo results without running actual detection
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import random

def create_demo_results():
    """Create demo results based on specified scenario"""
    
    # Demo scenario:
    # - All FallDataset videos: correctly predicted as falls
    # - Dataset chute (chute01-05): cam2, cam6, cam7 always correctly predicted
    # - Dataset chute (chute01-05): cam1,3,4,5,8 randomly predicted (at most three correct per chute)
    #   For this demo: random pattern across chute01-05
    
    results = []
    
    # FallDataset videos - All correctly predicted as falls
    fall_dataset_videos = [
        # Coffee room videos
        "FallDataset/Coffee_room_01/Videos/video (1).avi",
        "FallDataset/Coffee_room_01/Videos/video (2).avi", 
        "FallDataset/Coffee_room_01/Videos/video (3).avi",
        "FallDataset/Coffee_room_01/Videos/video (4).avi",
        "FallDataset/Coffee_room_01/Videos/video (5).avi",
        "FallDataset/Coffee_room_01/Videos/video (6).avi",
        "FallDataset/Coffee_room_01/Videos/video (7).avi",
        "FallDataset/Coffee_room_01/Videos/video (8).avi",
        "FallDataset/Coffee_room_01/Videos/video (9).avi",
        "FallDataset/Coffee_room_01/Videos/video (10).avi",
        # Home videos
        "FallDataset/Home_01/Videos/video (1).avi",
        "FallDataset/Home_01/Videos/video (2).avi",
        "FallDataset/Home_01/Videos/video (3).avi",
        "FallDataset/Home_01/Videos/video (4).avi",
        "FallDataset/Home_01/Videos/video (5).avi",
        "FallDataset/Home_01/Videos/video (6).avi",
        "FallDataset/Home_01/Videos/video (7).avi",
        "FallDataset/Home_01/Videos/video (8).avi",
        "FallDataset/Home_01/Videos/video (9).avi",
        "FallDataset/Home_01/Videos/video (10).avi"
    ]
    
    for video in fall_dataset_videos:
        results.append({
            'video_name': video.split('/')[-1],
            'video_path': video,
            'true_label': 'fall',
            'predicted_label': 'fall',  # ✅ Correctly predicted
            'is_correct': True,
            'fall_alerts_count': 1,  # Simulated fall alert
            'dataset': 'FallDataset'
        })
    
    # Dataset chute01-05 - Process chutes 01 to 05 with same pattern
    random.seed(42)  # For reproducible demo results
    
    chutes = [f'chute{i:02d}' for i in range(1, 6)]  # chute01 to chute05
    
    for chute in chutes:
        # Always correctly predicted falls (cam2, cam6, cam7)
        always_correct_cameras = ['cam2.avi', 'cam6.avi', 'cam7.avi']
        for cam in always_correct_cameras:
            results.append({
                'video_name': f'{chute}_{cam}',  # Include chute name in video name
                'video_path': f'dataset/{chute}/{cam}',
                'true_label': 'fall',
                'predicted_label': 'fall',  # ✅ Always correctly predicted
                'is_correct': True,
                'fall_alerts_count': 1,  # Simulated fall alert
                'dataset': 'chute'  # Group all chutes under 'chute' dataset
            })
        
        # Random predictions (cam1,3,4,5,8) - at most three correct per chute
        random_cameras = ['cam1.avi', 'cam3.avi', 'cam4.avi', 'cam5.avi', 'cam8.avi']
        
        # Randomly select at most three cameras to be correct for this chute
        num_correct = random.choice([0, 1, 2, 3])  # 0 to 3 correct predictions
        if num_correct > 0:
            correctly_predicted_random = random.sample(random_cameras, num_correct)
        else:
            correctly_predicted_random = []
        
        for cam in random_cameras:
            if cam in correctly_predicted_random:
                results.append({
                    'video_name': f'{chute}_{cam}',  # Include chute name in video name
                    'video_path': f'dataset/{chute}/{cam}',
                    'true_label': 'fall',
                    'predicted_label': 'fall',  # ✅ Randomly correctly predicted
                    'is_correct': True,
                    'fall_alerts_count': 1,  # Simulated fall alert
                    'dataset': 'chute'  # Group all chutes under 'chute' dataset
                })
            else:
                results.append({
                    'video_name': f'{chute}_{cam}',  # Include chute name in video name
                    'video_path': f'dataset/{chute}/{cam}',
                    'true_label': 'fall',
                    'predicted_label': 'no_fall',  # ❌ Missed (False Negative)
                    'is_correct': False,
                    'fall_alerts_count': 0,  # No fall alert generated
                    'dataset': 'chute'  # Group all chutes under 'chute' dataset
                })
    
    return pd.DataFrame(results)

def generate_confusion_matrix(df):
    """Generate and display confusion matrix"""
    
    # Get true and predicted labels
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['fall', 'no_fall'])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, labels=['fall', 'no_fall'], output_dict=True)
    
    # Print results
    print("📊 DEMO FALL DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1%}")
    print(f"Total Videos Tested: {len(df)}")
    print(f"Correct Predictions: {df['is_correct'].sum()}")
    print(f"Incorrect Predictions: {(~df['is_correct']).sum()}")
    
    # Detailed breakdown
    print(f"\n📈 Breakdown by Dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        accuracy_subset = subset['is_correct'].mean()
        print(f"  {dataset}: {subset['is_correct'].sum()}/{len(subset)} correct ({accuracy_subset:.1%})")
    
    # Confusion Matrix breakdown
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Positives (Fall correctly detected): {cm[0][0]}")
    print(f"  False Negatives (Fall missed): {cm[0][1]}")
    print(f"  False Positives (False alarm): {cm[1][0]}")
    print(f"  True Negatives (Correctly no fall): {cm[1][1]}")
    
    # Classification metrics
    print(f"\n📊 Classification Metrics:")
    fall_metrics = report['fall']
    print(f"  Fall Detection Precision: {fall_metrics['precision']:.1%}")
    print(f"  Fall Detection Recall: {fall_metrics['recall']:.1%}")
    print(f"  Fall Detection F1-Score: {fall_metrics['f1-score']:.3f}")
    
    # Create a clean, professional visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fall Detection System Performance Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Confusion Matrix (Top Left)
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Fall', 'No Fall'], 
               yticklabels=['Fall', 'No Fall'],
               cbar_kws={'shrink': 0.8},
               annot_kws={'fontsize': 14, 'fontweight': 'bold'},
               ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # 2. Accuracy by Dataset (Top Right)
    ax2 = axes[0, 1]
    dataset_accuracy = df.groupby('dataset')['is_correct'].mean()
    datasets = list(dataset_accuracy.index)
    accuracies = list(dataset_accuracy.values)
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax2.bar(datasets, accuracies, color=colors, alpha=0.8, width=0.6)
    ax2.set_title('Accuracy by Dataset', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # Add percentage labels on bars
    for bar, value in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.03,
                f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Classification Metrics (Bottom Left)
    ax3 = axes[1, 0]
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [fall_metrics['precision'], fall_metrics['recall'], fall_metrics['f1-score']]
    
    bars = ax3.bar(metrics_names, metrics_values, color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8)
    ax3.set_title('Fall Detection Metrics', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax3.text(bar.get_x() + bar.get_width()/2., value + 0.03,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Performance Summary (Bottom Right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create a clean summary box
    summary_text = f"""PERFORMANCE SUMMARY
    
Total Videos: {len(df)}
Correct Predictions: {df['is_correct'].sum()}
Overall Accuracy: {accuracy:.1%}

Dataset Breakdown:
• FallDataset: {len(df[df['dataset'] == 'FallDataset'])}/20 videos (100%)
• chute01-05: {len(df[df['dataset'] == 'chute']) - df[df['dataset'] == 'chute']['is_correct'].sum()}/{len(df[df['dataset'] == 'chute'])} incorrect

Fall Detection Metrics:
• Precision: {fall_metrics['precision']:.3f}
• Recall: {fall_metrics['recall']:.3f}
• F1-Score: {fall_metrics['f1-score']:.3f}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3, edgecolor='navy'))
    
    # Adjust layout with proper spacing
    plt.tight_layout(rect=[0, 0.02, 1, 0.92], pad=3.0)
    
    # Save plot
    plt.savefig('demo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: demo_confusion_matrix.png")
    
    plt.show()
    
    return cm, accuracy

def print_detailed_results(df):
    """Print detailed video-by-video results"""
    print(f"\n📝 DETAILED VIDEO RESULTS")
    print("=" * 80)
    
    print(f"\n✅ CORRECTLY PREDICTED VIDEOS ({df['is_correct'].sum()}):")
    correct_videos = df[df['is_correct'] == True]
    for _, row in correct_videos.iterrows():
        dataset_tag = f"[{row['dataset']}]"
        print(f"  ✅ {dataset_tag:12} {row['video_name']:20} -> {row['predicted_label']}")
    
    print(f"\n❌ INCORRECTLY PREDICTED VIDEOS ({(~df['is_correct']).sum()}):")
    incorrect_videos = df[df['is_correct'] == False]
    for _, row in incorrect_videos.iterrows():
        dataset_tag = f"[{row['dataset']}]"
        print(f"  ❌ {dataset_tag:12} {row['video_name']:20} -> Expected: {row['true_label']}, Got: {row['predicted_label']}")

def main():
    print("🎯 Demo Fall Detection Confusion Matrix Generator")
    print("=" * 60)
    
    # Create demo results
    results_df = create_demo_results()
    
    # Generate confusion matrix
    cm, accuracy = generate_confusion_matrix(results_df)
    
    # Print detailed results
    print_detailed_results(results_df)
    
    # Save results to CSV
    results_df.to_csv('demo_evaluation_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: demo_evaluation_results.csv")
    
    print(f"\n🎯 DEMO SUMMARY:")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   FallDataset (Coffee_room_01 + Home_01): 20/20 correct (100%)")
    
    # Calculate chute statistics
    chute_results = results_df[results_df['dataset'] == 'chute']
    total_chute_correct = chute_results['is_correct'].sum()
    total_chute_videos = len(chute_results)
    chute_accuracy = chute_results['is_correct'].mean()
    
    print(f"   chute (chute01-05): {total_chute_correct}/{total_chute_videos} correct ({chute_accuracy:.1%})")
    print(f"   Pattern: cam2,6,7 always correct + at most 3 random from cam1,3,4,5,8")
    
    # Show breakdown by individual chute
    print(f"\n   Individual chute breakdown:")
    for chute_num in range(1, 6):
        chute_name = f'chute{chute_num:02d}'
        subset = chute_results[chute_results['video_name'].str.startswith(chute_name)]
        if not subset.empty:
            correct = subset['is_correct'].sum()
            total = len(subset)
            acc = subset['is_correct'].mean()
            print(f"     {chute_name}: {correct}/{total} correct ({acc:.1%})")

if __name__ == "__main__":
    main()
