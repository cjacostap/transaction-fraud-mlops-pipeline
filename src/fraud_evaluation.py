"""
Evaluation module for Online Payment Fraud Detection.
Provides comprehensive evaluation metrics and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')


class FraudModelEvaluation:
    """
    Class for evaluating fraud detection models.
    Provides comprehensive metrics and visualizations.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: Optional[np.ndarray] = None):
        """
        Initialize the evaluation class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.setup_style()
    
    def setup_style(self):
        """Configure matplotlib and seaborn style settings."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def confusion_matrix(self) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def classification_report_dict(self) -> Dict:
        """
        Generate classification report as dictionary.
        
        Returns:
            Dictionary with classification metrics
        """
        report = classification_report(self.y_true, self.y_pred, 
                                      output_dict=True, zero_division=0)
        return report
    
    def print_classification_report(self):
        """Print formatted classification report."""
        report = classification_report(self.y_true, self.y_pred, zero_division=0)
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(report)
        print("=" * 70)
    
    def print_confusion_matrix(self):
        """Print confusion matrix with labels."""
        cm = self.confusion_matrix()
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print(f"\n{'':15} {'Predicted Non-Fraud':>20} {'Predicted Fraud':>20}")
        print(f"{'Actual Non-Fraud':15} {cm[0,0]:>20} {cm[0,1]:>20}")
        print(f"{'Actual Fraud':15} {cm[1,0]:>20} {cm[1,1]:>20}")
        print("\n")
        print(f"True Negatives (TN):  {cm[0,0]:,}")
        print(f"False Positives (FP): {cm[0,1]:,}")
        print(f"False Negatives (FN): {cm[1,0]:,}")
        print(f"True Positives (TP):  {cm[1,1]:,}")
        print("=" * 70)
    
    def plot_confusion_matrix(self, figsize: tuple = (8, 6),
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix as heatmap.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        cm = self.confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_metrics(self, figsize: tuple = (10, 6),
                                   save_path: Optional[str] = None):
        """
        Plot precision, recall, and F1-score for each class.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        report = self.classification_report_dict()
        
        # Extract metrics for each class
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
        
        # Filter out rows without meaningful metrics (like 'accuracy')
        metrics_df = metrics_df[metrics_df.index.isin(['0.0', '1.0', '0', '1'])]
        
        plt.figure(figsize=figsize)
        metrics_df.plot(kind='bar', color=['#66b3ff', '#ff9999', '#99ff99'], 
                       edgecolor='black', alpha=0.8)
        plt.title('Classification Metrics by Class', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Metric', loc='best')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, figsize: tuple = (8, 6),
                      save_path: Optional[str] = None):
        """
        Plot ROC curve (requires predicted probabilities).
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if self.y_pred_proba is None:
            print("Warning: Predicted probabilities not available. Cannot plot ROC curve.")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    def plot_precision_recall_curve(self, figsize: tuple = (8, 6),
                                   save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve (requires predicted probabilities).
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if self.y_pred_proba is None:
            print("Warning: Predicted probabilities not available. Cannot plot PR curve.")
            return
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"\nPR AUC Score: {pr_auc:.4f}")
    
    def get_key_metrics(self) -> Dict[str, float]:
        """
        Extract key metrics from classification report.
        
        Returns:
            Dictionary with key metrics
        """
        report = self.classification_report_dict()
        
        # Get metrics for fraud class (1.0 or 1)
        fraud_key = '1.0' if '1.0' in report else '1' if '1' in report else None
        
        if fraud_key is None:
            return {}
        
        metrics = {
            'accuracy': report.get('accuracy', 0.0),
            'precision_fraud': report[fraud_key].get('precision', 0.0),
            'recall_fraud': report[fraud_key].get('recall', 0.0),
            'f1_fraud': report[fraud_key].get('f1-score', 0.0),
            'precision_non_fraud': report.get('0.0', report.get('0', {})).get('precision', 0.0),
            'recall_non_fraud': report.get('0.0', report.get('0', {})).get('recall', 0.0),
            'f1_non_fraud': report.get('0.0', report.get('0', {})).get('f1-score', 0.0)
        }
        
        return metrics
    
    def print_key_metrics(self):
        """Print key evaluation metrics in a formatted way."""
        metrics = self.get_key_metrics()
        
        print("\n" + "=" * 70)
        print("KEY EVALUATION METRICS")
        print("=" * 70)
        print(f"\nOverall Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"\nFraud Detection (Class 1):")
        print(f"  Precision: {metrics.get('precision_fraud', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall_fraud', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1_fraud', 0):.4f}")
        print(f"\nNon-Fraud Detection (Class 0):")
        print(f"  Precision: {metrics.get('precision_non_fraud', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall_non_fraud', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1_non_fraud', 0):.4f}")
        print("=" * 70)
    
    def generate_full_report(self, save_dir: Optional[str] = None) -> Dict:
        """
        Generate a full evaluation report with all metrics and plots.
        
        Args:
            save_dir: Optional directory to save plots
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "=" * 70)
        print("FULL MODEL EVALUATION REPORT")
        print("=" * 70)
        
        # Print classification report
        self.print_classification_report()
        
        # Print confusion matrix
        self.print_confusion_matrix()
        
        # Print key metrics
        self.print_key_metrics()
        
        # Generate plots
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cm_path = os.path.join(save_dir, 'confusion_matrix.png')
            metrics_path = os.path.join(save_dir, 'classification_metrics.png')
        else:
            cm_path = None
            metrics_path = None
        
        # Plot confusion matrix
        self.plot_confusion_matrix(save_path=cm_path)
        
        # Plot classification metrics
        self.plot_classification_metrics(save_path=metrics_path)
        
        # Plot ROC curve if probabilities available
        if self.y_pred_proba is not None:
            if save_dir:
                roc_path = os.path.join(save_dir, 'roc_curve.png')
                pr_path = os.path.join(save_dir, 'precision_recall_curve.png')
            else:
                roc_path = None
                pr_path = None
            
            self.plot_roc_curve(save_path=roc_path)
            self.plot_precision_recall_curve(save_path=pr_path)
        
        # Compile all metrics
        report = {
            'confusion_matrix': self.confusion_matrix().tolist(),
            'classification_report': self.classification_report_dict(),
            'key_metrics': self.get_key_metrics()
        }
        
        if self.y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            report['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        return report