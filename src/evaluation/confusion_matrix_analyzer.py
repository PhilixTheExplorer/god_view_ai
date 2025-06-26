"""
Confusion Matrix Analyzer for Fall Detection System
Provides comprehensive evaluation metrics for fall detection performance
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2


class ConfusionMatrixAnalyzer:
    """
    Analyzes fall detection performance using confusion matrices and various metrics.
    
    This class provides:
    - Binary confusion matrix (Fall vs No-Fall)
    - Multi-class confusion matrix (Standing, Sitting, Lying, Fall)
    - Performance metrics (Precision, Recall, F1-Score, Accuracy)
    - Visualization and reporting
    """
    
    def __init__(self, save_dir: str = "evaluation_results"):
        """
        Initialize the confusion matrix analyzer.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Store predictions and ground truth
        self.predictions = []
        self.ground_truth = []
        self.timestamps = []
        self.frame_numbers = []
        self.confidence_scores = []
        
        # For detailed analysis
        self.posture_predictions = []
        self.posture_ground_truth = []
        
        # For frame-by-frame analysis
        self.frame_data = []
        
    def add_prediction(self, 
                      predicted_fall: bool,
                      actual_fall: bool,
                      predicted_posture: str,
                      actual_posture: str,
                      confidence: float = 1.0,
                      frame_number: int = 0,
                      timestamp: Optional[datetime] = None):
        """
        Add a single prediction result.
        
        Args:
            predicted_fall: Whether system predicted a fall
            actual_fall: Whether a fall actually occurred
            predicted_posture: Predicted posture ('standing', 'sitting', 'lying')
            actual_posture: Actual posture
            confidence: Confidence score of the prediction
            frame_number: Frame number in video
            timestamp: Timestamp of the prediction
        """
        self.predictions.append(predicted_fall)
        self.ground_truth.append(actual_fall)
        self.posture_predictions.append(predicted_posture)
        self.posture_ground_truth.append(actual_posture)
        self.confidence_scores.append(confidence)
        self.frame_numbers.append(frame_number)
        self.timestamps.append(timestamp or datetime.now())
        
        # Store frame data for detailed analysis
        self.frame_data.append({
            'frame_number': frame_number,
            'predicted_fall': predicted_fall,
            'actual_fall': actual_fall,
            'predicted_posture': predicted_posture,
            'actual_posture': actual_posture,
            'confidence': confidence,
            'timestamp': timestamp or datetime.now()
        })
    
    def add_batch_predictions(self, 
                            predicted_falls: List[bool],
                            actual_falls: List[bool],
                            predicted_postures: List[str],
                            actual_postures: List[str],
                            confidences: Optional[List[float]] = None,
                            frame_numbers: Optional[List[int]] = None):
        """
        Add batch predictions for efficiency.
        
        Args:
            predicted_falls: List of fall predictions
            actual_falls: List of actual fall labels
            predicted_postures: List of posture predictions
            actual_postures: List of actual posture labels
            confidences: List of confidence scores
            frame_numbers: List of frame numbers
        """
        if confidences is None:
            confidences = [1.0] * len(predicted_falls)
        if frame_numbers is None:
            frame_numbers = list(range(len(predicted_falls)))
            
        for i in range(len(predicted_falls)):
            self.add_prediction(
                predicted_falls[i],
                actual_falls[i], 
                predicted_postures[i],
                actual_postures[i],
                confidences[i],
                frame_numbers[i]
            )
    
    def generate_fall_detection_confusion_matrix(self, save_plot: bool = True) -> np.ndarray:
        """
        Generate confusion matrix for binary fall detection.
        
        Args:
            save_plot: Whether to save the visualization
            
        Returns:
            2x2 confusion matrix array
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions available. Add predictions first.")
        
        # Generate confusion matrix
        cm = confusion_matrix(self.ground_truth, self.predictions)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fall', 'Fall'],
                   yticklabels=['No Fall', 'Fall'])
        plt.title('Fall Detection Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add performance metrics as text
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        precision = precision_score(self.ground_truth, self.predictions, zero_division=0)
        recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        plt.text(2.5, 0.5, metrics_text, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(self.save_dir / f'fall_detection_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Fall detection confusion matrix saved to {self.save_dir}")
        
        plt.show()
        return cm
    
    def generate_posture_confusion_matrix(self, save_plot: bool = True) -> np.ndarray:
        """
        Generate confusion matrix for posture classification.
        
        Args:
            save_plot: Whether to save the visualization
            
        Returns:
            Confusion matrix array for postures
        """
        if len(self.posture_predictions) == 0:
            raise ValueError("No posture predictions available.")
        
        # Get unique postures
        unique_postures = sorted(list(set(self.posture_predictions + self.posture_ground_truth)))
        
        # Generate confusion matrix
        cm = confusion_matrix(self.posture_ground_truth, self.posture_predictions, labels=unique_postures)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=unique_postures,
                   yticklabels=unique_postures)
        plt.title('Posture Classification Confusion Matrix')
        plt.xlabel('Predicted Posture')
        plt.ylabel('Actual Posture')
        
        # Calculate per-class metrics
        report = classification_report(self.posture_ground_truth, self.posture_predictions, 
                                     labels=unique_postures, output_dict=True, zero_division=0)
        
        # Add overall accuracy
        accuracy = accuracy_score(self.posture_ground_truth, self.posture_predictions)
        plt.text(len(unique_postures) + 0.5, len(unique_postures)/2, 
                f'Overall Accuracy: {accuracy:.3f}', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(self.save_dir / f'posture_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Posture confusion matrix saved to {self.save_dir}")
        
        plt.show()
        return cm
    
    def generate_detailed_report(self, save_report: bool = True) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_report: Whether to save the report to file
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions available.")
        
        # Fall detection metrics
        fall_accuracy = accuracy_score(self.ground_truth, self.predictions)
        fall_precision = precision_score(self.ground_truth, self.predictions, zero_division=0)
        fall_recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        fall_f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)
        
        # Posture classification metrics
        posture_accuracy = accuracy_score(self.posture_ground_truth, self.posture_predictions)
        posture_report = classification_report(self.posture_ground_truth, self.posture_predictions, 
                                             output_dict=True, zero_division=0)
        
        # Generate confusion matrices
        fall_cm = confusion_matrix(self.ground_truth, self.predictions)
        posture_cm = confusion_matrix(self.posture_ground_truth, self.posture_predictions)
        
        # Calculate additional metrics
        total_frames = len(self.predictions)
        total_falls_detected = sum(self.predictions)
        total_actual_falls = sum(self.ground_truth)
        
        # False positive analysis
        false_positives = sum(1 for pred, actual in zip(self.predictions, self.ground_truth) 
                            if pred and not actual)
        false_negatives = sum(1 for pred, actual in zip(self.predictions, self.ground_truth) 
                            if not pred and actual)
        
        # Compile report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_frames': total_frames,
                'total_predictions': len(self.predictions),
                'actual_falls': total_actual_falls,
                'detected_falls': total_falls_detected,
            },
            'fall_detection_metrics': {
                'accuracy': float(fall_accuracy),
                'precision': float(fall_precision),
                'recall': float(fall_recall),
                'f1_score': float(fall_f1),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'confusion_matrix': fall_cm.tolist()
            },
            'posture_classification_metrics': {
                'accuracy': float(posture_accuracy),
                'detailed_report': posture_report,
                'confusion_matrix': posture_cm.tolist()
            },
            'confidence_analysis': {
                'mean_confidence': float(np.mean(self.confidence_scores)),
                'std_confidence': float(np.std(self.confidence_scores)),
                'min_confidence': float(np.min(self.confidence_scores)),
                'max_confidence': float(np.max(self.confidence_scores))
            }
        }
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.save_dir / f'evaluation_report_{timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Detailed evaluation report saved to {report_path}")
        
        return report
    
    def plot_performance_over_time(self, save_plot: bool = True):
        """
        Plot performance metrics over time/frames.
        
        Args:
            save_plot: Whether to save the plot
        """
        if len(self.frame_data) == 0:
            raise ValueError("No frame data available.")
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame(self.frame_data)
        
        # Calculate rolling accuracy (window of 100 frames)
        window_size = min(100, len(df) // 10)
        df['correct_fall'] = df['predicted_fall'] == df['actual_fall']
        df['correct_posture'] = df['predicted_posture'] == df['actual_posture']
        df['rolling_fall_accuracy'] = df['correct_fall'].rolling(window=window_size, min_periods=1).mean()
        df['rolling_posture_accuracy'] = df['correct_posture'].rolling(window=window_size, min_periods=1).mean()
        
        # Create subplot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Rolling accuracy over time
        ax1.plot(df['frame_number'], df['rolling_fall_accuracy'], label='Fall Detection Accuracy', color='red')
        ax1.plot(df['frame_number'], df['rolling_posture_accuracy'], label='Posture Classification Accuracy', color='blue')
        ax1.set_ylabel('Rolling Accuracy')
        ax1.set_title(f'Performance Over Time (Window Size: {window_size} frames)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence scores over time
        ax2.plot(df['frame_number'], df['confidence'], alpha=0.7, color='green')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Scores Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fall detections over time
        fall_frames = df[df['predicted_fall'] == True]['frame_number']
        actual_fall_frames = df[df['actual_fall'] == True]['frame_number']
        
        ax3.scatter(fall_frames, [1] * len(fall_frames), label='Predicted Falls', color='red', alpha=0.7, s=30)
        ax3.scatter(actual_fall_frames, [0.5] * len(actual_fall_frames), label='Actual Falls', color='orange', alpha=0.7, s=30)
        ax3.set_ylabel('Fall Events')
        ax3.set_xlabel('Frame Number')
        ax3.set_title('Fall Detections Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.5)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(self.save_dir / f'performance_over_time_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Performance over time plot saved to {self.save_dir}")
        
        plt.show()
    
    def analyze_false_positives_negatives(self, save_analysis: bool = True) -> Dict:
        """
        Analyze false positives and false negatives in detail.
        
        Args:
            save_analysis: Whether to save the analysis
            
        Returns:
            Dictionary with detailed analysis
        """
        if len(self.frame_data) == 0:
            raise ValueError("No frame data available.")
        
        df = pd.DataFrame(self.frame_data)
        
        # Identify false positives and negatives
        false_positives = df[(df['predicted_fall'] == True) & (df['actual_fall'] == False)]
        false_negatives = df[(df['predicted_fall'] == False) & (df['actual_fall'] == True)]
        true_positives = df[(df['predicted_fall'] == True) & (df['actual_fall'] == True)]
        true_negatives = df[(df['predicted_fall'] == False) & (df['actual_fall'] == False)]
        
        analysis = {
            'false_positives': {
                'count': len(false_positives),
                'percentage': len(false_positives) / len(df) * 100,
                'posture_distribution': false_positives['predicted_posture'].value_counts().to_dict(),
                'confidence_stats': {
                    'mean': false_positives['confidence'].mean() if len(false_positives) > 0 else 0,
                    'std': false_positives['confidence'].std() if len(false_positives) > 0 else 0,
                }
            },
            'false_negatives': {
                'count': len(false_negatives),
                'percentage': len(false_negatives) / len(df) * 100,
                'posture_distribution': false_negatives['actual_posture'].value_counts().to_dict(),
                'confidence_stats': {
                    'mean': false_negatives['confidence'].mean() if len(false_negatives) > 0 else 0,
                    'std': false_negatives['confidence'].std() if len(false_negatives) > 0 else 0,
                }
            },
            'true_positives': {
                'count': len(true_positives),
                'percentage': len(true_positives) / len(df) * 100,
            },
            'true_negatives': {
                'count': len(true_negatives),
                'percentage': len(true_negatives) / len(df) * 100,
            }
        }
        
        if save_analysis:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_path = self.save_dir / f'false_positive_negative_analysis_{timestamp}.json'
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"✅ False positive/negative analysis saved to {analysis_path}")
        
        return analysis
    
    def create_comprehensive_visualization(self, save_plot: bool = True):
        """
        Create a comprehensive visualization with multiple plots.
        
        Args:
            save_plot: Whether to save the visualization
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Fall Detection Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        fall_cm = confusion_matrix(self.ground_truth, self.predictions)
        sns.heatmap(fall_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Fall', 'Fall'],
                   yticklabels=['No Fall', 'Fall'])
        plt.title('Fall Detection Confusion Matrix')
        
        # 2. Posture Confusion Matrix
        ax2 = plt.subplot(2, 3, 2)
        unique_postures = sorted(list(set(self.posture_predictions + self.posture_ground_truth)))
        posture_cm = confusion_matrix(self.posture_ground_truth, self.posture_predictions, labels=unique_postures)
        sns.heatmap(posture_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=unique_postures,
                   yticklabels=unique_postures)
        plt.title('Posture Classification CM')
        
        # 3. Performance Metrics Bar Chart
        ax3 = plt.subplot(2, 3, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fall_values = [
            accuracy_score(self.ground_truth, self.predictions),
            precision_score(self.ground_truth, self.predictions, zero_division=0),
            recall_score(self.ground_truth, self.predictions, zero_division=0),
            f1_score(self.ground_truth, self.predictions, zero_division=0)
        ]
        posture_values = [
            accuracy_score(self.posture_ground_truth, self.posture_predictions),
            precision_score(self.posture_ground_truth, self.posture_predictions, average='weighted', zero_division=0),
            recall_score(self.posture_ground_truth, self.posture_predictions, average='weighted', zero_division=0),
            f1_score(self.posture_ground_truth, self.posture_predictions, average='weighted', zero_division=0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, fall_values, width, label='Fall Detection', color='red', alpha=0.7)
        ax3.bar(x + width/2, posture_values, width, label='Posture Classification', color='blue', alpha=0.7)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence Distribution
        ax4 = plt.subplot(2, 3, 4)
        plt.hist(self.confidence_scores, bins=20, alpha=0.7, color='green')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # 5. Fall vs No-Fall Distribution
        ax5 = plt.subplot(2, 3, 5)
        fall_counts = [sum(1 for x in self.ground_truth if not x), sum(self.ground_truth)]
        pred_counts = [sum(1 for x in self.predictions if not x), sum(self.predictions)]
        
        x = ['No Fall', 'Fall']
        width = 0.35
        ax5.bar([i - width/2 for i in range(len(x))], fall_counts, width, label='Ground Truth', alpha=0.7)
        ax5.bar([i + width/2 for i in range(len(x))], pred_counts, width, label='Predictions', alpha=0.7)
        ax5.set_xlabel('Categories')
        ax5.set_ylabel('Count')
        ax5.set_title('Fall vs No-Fall Distribution')
        ax5.set_xticks(range(len(x)))
        ax5.set_xticklabels(x)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error Analysis
        ax6 = plt.subplot(2, 3, 6)
        df = pd.DataFrame(self.frame_data)
        false_positives = len(df[(df['predicted_fall'] == True) & (df['actual_fall'] == False)])
        false_negatives = len(df[(df['predicted_fall'] == False) & (df['actual_fall'] == True)])
        true_positives = len(df[(df['predicted_fall'] == True) & (df['actual_fall'] == True)])
        true_negatives = len(df[(df['predicted_fall'] == False) & (df['actual_fall'] == False)])
        
        categories = ['True\nPositives', 'True\nNegatives', 'False\nPositives', 'False\nNegatives']
        values = [true_positives, true_negatives, false_positives, false_negatives]
        colors = ['green', 'blue', 'orange', 'red']
        
        ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_ylabel('Count')
        ax6.set_title('Error Analysis')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(self.save_dir / f'comprehensive_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"✅ Comprehensive evaluation visualization saved to {self.save_dir}")
        
        plt.show()
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if len(self.predictions) == 0:
            print("❌ No predictions available for analysis.")
            return
        
        # Calculate metrics
        fall_accuracy = accuracy_score(self.ground_truth, self.predictions)
        fall_precision = precision_score(self.ground_truth, self.predictions, zero_division=0)
        fall_recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        fall_f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)
        
        posture_accuracy = accuracy_score(self.posture_ground_truth, self.posture_predictions)
        
        # Calculate false positives and negatives
        df = pd.DataFrame(self.frame_data)
        false_positives = len(df[(df['predicted_fall'] == True) & (df['actual_fall'] == False)])
        false_negatives = len(df[(df['predicted_fall'] == False) & (df['actual_fall'] == True)])
        
        print("=" * 60)
        print("🔍 FALL DETECTION EVALUATION SUMMARY")
        print("=" * 60)
        print(f"📊 Dataset Info:")
        print(f"   Total Frames: {len(self.predictions)}")
        print(f"   Actual Falls: {sum(self.ground_truth)}")
        print(f"   Detected Falls: {sum(self.predictions)}")
        print()
        print(f"🎯 Fall Detection Performance:")
        print(f"   Accuracy:  {fall_accuracy:.3f}")
        print(f"   Precision: {fall_precision:.3f}")
        print(f"   Recall:    {fall_recall:.3f}")
        print(f"   F1-Score:  {fall_f1:.3f}")
        print()
        print(f"🏃 Posture Classification Performance:")
        print(f"   Accuracy:  {posture_accuracy:.3f}")
        print()
        print(f"❌ Error Analysis:")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        print()
        print(f"📈 Confidence Statistics:")
        print(f"   Mean: {np.mean(self.confidence_scores):.3f}")
        print(f"   Std:  {np.std(self.confidence_scores):.3f}")
        print("=" * 60)


def create_sample_evaluation():
    """
    Create a sample evaluation to demonstrate usage.
    This can be used as a template for real evaluations.
    """
    print("🔍 Creating sample confusion matrix evaluation...")
    
    # Create analyzer
    analyzer = ConfusionMatrixAnalyzer("evaluation_results")
    
    # Sample data (replace with real data)
    np.random.seed(42)  # For reproducible results
    
    # Simulate video frames with different scenarios
    scenarios = [
        # Normal activity (no falls)
        ("standing", False, 50),
        ("sitting", False, 30),
        ("lying", False, 40),  # Sleeping/resting
        
        # Fall scenarios
        ("lying", True, 15),   # Falls resulting in lying position
        
        # Some misclassifications for realistic evaluation
        ("standing", False, 5),  # False positives
        ("sitting", True, 3),    # Missed falls
    ]
    
    frame_count = 0
    for actual_posture, is_fall, count in scenarios:
        for i in range(count):
            # Add some noise to predictions
            if np.random.random() < 0.1:  # 10% classification error
                postures = ["standing", "sitting", "lying"]
                predicted_posture = np.random.choice([p for p in postures if p != actual_posture])
            else:
                predicted_posture = actual_posture
            
            # Fall prediction based on posture + some noise
            if is_fall:
                predicted_fall = np.random.random() < 0.9  # 90% detection rate for actual falls
            else:
                predicted_fall = np.random.random() < 0.05  # 5% false positive rate
            
            # Random confidence
            confidence = np.random.uniform(0.6, 0.95)
            
            analyzer.add_prediction(
                predicted_fall=predicted_fall,
                actual_fall=is_fall,
                predicted_posture=predicted_posture,
                actual_posture=actual_posture,
                confidence=confidence,
                frame_number=frame_count
            )
            frame_count += 1
    
    return analyzer


if __name__ == "__main__":
    # Create sample evaluation
    analyzer = create_sample_evaluation()
    
    # Generate all visualizations and reports
    print("\n📊 Generating evaluation reports...")
    
    # Print summary
    analyzer.print_summary()
    
    # Generate confusion matrices
    analyzer.generate_fall_detection_confusion_matrix()
    analyzer.generate_posture_confusion_matrix()
    
    # Generate comprehensive visualization
    analyzer.create_comprehensive_visualization()
    
    # Generate detailed report
    report = analyzer.generate_detailed_report()
    
    # Analyze false positives and negatives
    fp_fn_analysis = analyzer.analyze_false_positives_negatives()
    
    # Plot performance over time
    analyzer.plot_performance_over_time()
    
    print("\n✅ Evaluation complete! Check the 'evaluation_results' folder for all outputs.")
