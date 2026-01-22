"""
Exploratory Data Analysis module for Online Payment Fraud Detection.
This module provides comprehensive EDA functionality based on the Online_Payment_Fraud_Detection notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path


class FraudEDA:
    """
    Class for performing exploratory data analysis on fraud detection dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FraudEDA with a DataFrame.
        
        Args:
            df: DataFrame containing the fraud detection data
        """
        self.df = df.copy()
        self.setup_style()
    
    def setup_style(self):
        """Configure matplotlib and seaborn style settings."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def basic_info(self) -> dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary with shape, info, and missing values
        """
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        return info
    
    def print_basic_info(self):
        """Print basic dataset information."""
        print("=" * 60)
        print("DATASET BASIC INFORMATION")
        print("=" * 60)
        print(f"\nShape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        print(f"\nColumn Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found!")
        else:
            print(missing[missing > 0])
        
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\n" + "=" * 60)
    
    def statistical_summary(self) -> pd.DataFrame:
        """Get statistical summary of numerical columns."""
        return self.df.describe()
    
    def value_distributions(self, columns: Optional[List[str]] = None) -> dict:
        """
        Analyze value distributions for specified columns.
        
        Args:
            columns: List of column names to analyze. If None, uses common columns.
            
        Returns:
            Dictionary with value counts for each column
        """
        if columns is None:
            columns = ['type', 'isFraud', 'isFlaggedFraud']
        
        distributions = {}
        for col in columns:
            if col in self.df.columns:
                distributions[col] = self.df[col].value_counts()
        
        return distributions
    
    def plot_feature_distribution(self, feature: str, bins: int = 10, 
                                  color: str = 'grey', figsize: tuple = (12, 6),
                                  save_path: Optional[str] = None):
        """
        Visualize the distribution of a specific feature with histogram.
        
        Args:
            feature: Column name to plot
            bins: Number of bins for histogram
            color: Color of the bars
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if feature not in self.df.columns:
            print(f"Warning: Column '{feature}' not found in dataset")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot histogram
        self.df[feature].plot(kind='hist', bins=bins, facecolor=color, 
                             edgecolor='black', alpha=0.7, label='Data Distribution')
        
        # Calculate mean and median
        mean_value = self.df[feature].mean()
        median_value = self.df[feature].median()
        
        # Add mean and median lines
        plt.axvline(mean_value, color='red', linestyle='dashed', 
                   linewidth=3, label=f'Mean: {mean_value:.2f}')
        plt.axvline(median_value, color='blue', linestyle='dashed', 
                   linewidth=3, label=f'Median: {median_value:.2f}')
        
        # Formatting
        plt.title(f'Distribution of {feature}', fontsize=16)
        plt.xlabel(f'{feature} Values', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # Set x-axis limits
        x_min = self.df[feature].min()
        x_max = self.df[feature].max()
        plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_fraud_by_type(self) -> pd.DataFrame:
        """
        Analyze fraud distribution by transaction type.
        
        Returns:
            Crosstab DataFrame showing fraud vs non-fraud by type
        """
        if 'type' not in self.df.columns or 'isFraud' not in self.df.columns:
            print("Warning: 'type' or 'isFraud' columns not found")
            return pd.DataFrame()
        
        crosstab = pd.crosstab(index=self.df['type'], columns=self.df['isFraud'])
        return crosstab
    
    def plot_fraud_by_type(self, figsize: tuple = (10, 6), save_path: Optional[str] = None):
        """
        Visualize fraud distribution by transaction type.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        crosstab = self.analyze_fraud_by_type()
        
        if crosstab.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Grouped bar chart with log scale
        crosstab.plot.bar(ax=axes[0], rot=0, color=['lightblue', 'salmon'], alpha=0.75)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Transaction Type', fontsize=12)
        axes[0].set_ylabel('Log(Count of Transactions)', fontsize=12)
        axes[0].set_title('Fraud vs Non-Fraud by Transaction Type', fontsize=14)
        axes[0].legend(title='Fraud Status', labels=['Non-Fraud', 'Fraud'])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fraud only, focused view
        if 1 in crosstab.columns:
            crosstab[1].plot.bar(ax=axes[1], rot=0, color='salmon', alpha=0.75)
            axes[1].set_xlabel('Transaction Type', fontsize=12)
            axes[1].set_ylabel('Count of Fraudulent Transactions', fontsize=12)
            axes[1].set_title('Fraudulent Transactions by Type', fontsize=14)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_fraud_percentages(self) -> dict:
        """
        Calculate fraud percentages for different transaction types.
        
        Returns:
            Dictionary with fraud percentages by type
        """
        crosstab = self.analyze_fraud_by_type()
        
        if crosstab.empty:
            return {}
        
        percentages = {}
        
        for trans_type in crosstab.index:
            if 1 in crosstab.columns:
                fraud_count = crosstab.loc[trans_type, 1]
                total_count = crosstab.loc[trans_type].sum()
                percentages[trans_type] = (fraud_count / total_count) * 100 if total_count > 0 else 0
        
        return percentages
    
    def plot_amount_by_fraud(self, figsize: tuple = (10, 6), save_path: Optional[str] = None):
        """
        Visualize transaction amount distribution by fraud status.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if 'amount' not in self.df.columns or 'isFraud' not in self.df.columns:
            print("Warning: 'amount' or 'isFraud' columns not found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Strip plot
        sns.stripplot(y=self.df['amount'], ax=axes[0], color='teal', 
                     jitter=True, size=1, alpha=0.3)
        axes[0].set_xlabel('Transaction Amount', fontsize=12)
        axes[0].set_ylabel('Amount', fontsize=12)
        axes[0].set_title('Distribution of Transaction Amounts', fontsize=14)
        
        # Plot 2: Boxplot by fraud status
        sns.boxplot(x='isFraud', y='amount', data=self.df, ax=axes[1], 
                   palette='Set2', showfliers=False, linewidth=2)
        axes[1].set_xlabel('Fraudulent Transaction (0 = Non-Fraud, 1 = Fraud)', fontsize=12)
        axes[1].set_ylabel('Transaction Amount', fontsize=12)
        axes[1].set_title('Distribution of Transaction Amount by Fraud Status', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_fraud_statistics(self) -> dict:
        """
        Calculate comprehensive fraud statistics.
        
        Returns:
            Dictionary with fraud statistics
        """
        if 'isFraud' not in self.df.columns:
            return {}
        
        total_transactions = len(self.df)
        fraud_transactions = (self.df['isFraud'] == 1).sum()
        fraud_percentage = (fraud_transactions / total_transactions) * 100
        
        stats = {
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'fraud_percentage': fraud_percentage
        }
        
        if 'isFlaggedFraud' in self.df.columns:
            flagged_fraud = ((self.df['isFraud'] == 1) & (self.df['isFlaggedFraud'] == 1)).sum()
            flagged_percentage = (flagged_fraud / fraud_transactions * 100) if fraud_transactions > 0 else 0
            unflagged_percentage = 100 - flagged_percentage
            
            stats.update({
                'flagged_fraud': flagged_fraud,
                'flagged_percentage': flagged_percentage,
                'unflagged_percentage': unflagged_percentage
            })
        
        return stats
    
    def plot_fraud_amount_distribution(self, bins: int = 30, 
                                      figsize: tuple = (12, 6),
                                      save_path: Optional[str] = None):
        """
        Plot distribution of fraudulent transaction amounts.
        
        Args:
            bins: Number of bins for histogram
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if 'isFraud' not in self.df.columns or 'amount' not in self.df.columns:
            print("Warning: Required columns not found")
            return
        
        fraud_amount = self.df[self.df['isFraud'] == 1]['amount'].copy()
        
        if len(fraud_amount) == 0:
            print("No fraudulent transactions found")
            return
        
        plt.figure(figsize=figsize)
        plt.hist(fraud_amount, bins=bins, color='tomato', edgecolor='black', alpha=0.7)
        
        mean_value = fraud_amount.mean()
        median_value = fraud_amount.median()
        
        plt.axvline(mean_value, color='blue', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {mean_value:.2f}')
        plt.axvline(median_value, color='green', linestyle='dashed', 
                   linewidth=2, label=f'Median: {median_value:.2f}')
        
        plt.title('Distribution of Fraudulent Transaction Amounts', fontsize=16)
        plt.xlabel('Transaction Amount', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_heatmap(self, figsize: tuple = (12, 8), 
                                save_path: Optional[str] = None):
        """
        Plot correlation heatmap for numerical features.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        numerical_df = self.df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                   annot=True, cmap='Blues', fmt='.3f', square=True)
        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_step_distribution(self, bins: int = 50, figsize: tuple = (15, 6),
                              save_path: Optional[str] = None):
        """
        Plot distribution of step feature (time unit).
        
        Args:
            bins: Number of bins
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        if 'step' not in self.df.columns:
            print("Warning: 'step' column not found")
            return
        
        plt.figure(figsize=figsize)
        sns.histplot(self.df['step'], bins=bins, color='lightcoral', kde=True)
        plt.title('Distribution of Step (Time Unit)', fontsize=14)
        plt.xlabel('Step (Time Unit)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of the EDA.
        
        Returns:
            String containing the summary report
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("FRAUD DETECTION - EXPLORATORY DATA ANALYSIS SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Basic info
        info = self.basic_info()
        report_lines.append(f"Dataset Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        report_lines.append("")
        
        # Fraud statistics
        fraud_stats = self.analyze_fraud_statistics()
        if fraud_stats:
            report_lines.append("FRAUD STATISTICS:")
            report_lines.append(f"  Total Transactions: {fraud_stats['total_transactions']:,}")
            report_lines.append(f"  Fraudulent Transactions: {fraud_stats['fraud_transactions']:,}")
            report_lines.append(f"  Fraud Percentage: {fraud_stats['fraud_percentage']:.2f}%")
            
            if 'flagged_fraud' in fraud_stats:
                report_lines.append(f"  Flagged Fraud: {fraud_stats['flagged_fraud']}")
                report_lines.append(f"  Flagged Percentage: {fraud_stats['flagged_percentage']:.3f}%")
                report_lines.append(f"  Unflagged Percentage: {fraud_stats['unflagged_percentage']:.3f}%")
            report_lines.append("")
        
        # Fraud by type
        fraud_percentages = self.calculate_fraud_percentages()
        if fraud_percentages:
            report_lines.append("FRAUD PERCENTAGE BY TRANSACTION TYPE:")
            for trans_type, pct in fraud_percentages.items():
                report_lines.append(f"  {trans_type}: {pct:.2f}%")
            report_lines.append("")
        
        # Amount statistics
        if 'amount' in self.df.columns:
            report_lines.append("TRANSACTION AMOUNT STATISTICS:")
            report_lines.append(f"  Minimum: {self.df['amount'].min():,.2f}")
            report_lines.append(f"  Maximum: {self.df['amount'].max():,.2f}")
            report_lines.append(f"  Mean: {self.df['amount'].mean():,.2f}")
            report_lines.append(f"  Median: {self.df['amount'].median():,.2f}")
            report_lines.append("")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def print_summary_report(self):
        """Print the comprehensive summary report."""
        print(self.generate_summary_report())