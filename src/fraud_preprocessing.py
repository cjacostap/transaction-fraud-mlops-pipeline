"""
Preprocessing module for Online Payment Fraud Detection.
Handles outliers, encoding, scaling, and class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class FraudPreprocessing:
    """
    Class for preprocessing fraud detection data.
    Includes outlier handling, encoding, scaling, and class imbalance handling.
    """
    
    def __init__(self):
        """Initialize the FraudPreprocessing class."""
        self.scaler = None
        self.imputer_num = None
        self.imputer_cat = None
        self.feature_columns = None
        self.outlier_thresholds = {}
    
    def handle_outliers_quantile(self, df: pd.DataFrame, 
                                 features: Optional[List[str]] = None,
                                 lower_percentile: float = 0.10,
                                 upper_percentile: float = 0.90) -> pd.DataFrame:
        """
        Handle outliers using quantile-based flooring and capping.
        
        Args:
            df: Input DataFrame
            features: List of features to process. If None, uses default financial features.
            lower_percentile: Lower percentile for flooring (default 0.10)
            upper_percentile: Upper percentile for capping (default 0.90)
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        if features is None:
            features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest']
        
        print(f"\nHandling outliers for features: {features}")
        print(f"Using percentiles: {lower_percentile:.0%} and {upper_percentile:.0%}")
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # Calculate thresholds
            lower = df[feature].quantile(lower_percentile)
            upper = df[feature].quantile(upper_percentile)
            
            # Store thresholds
            self.outlier_thresholds[feature] = {'lower': lower, 'upper': upper}
            
            # Apply flooring and capping
            df[feature] = np.where(df[feature] < lower, lower, df[feature])
            df[feature] = np.where(df[feature] > upper, upper, df[feature])
            
            # Calculate and print skewness
            skewness = df[feature].skew()
            print(f"  {feature}: Skewness = {skewness:.4f}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using imputation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed
        """
        df = df.copy()
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("\nNo missing values found in the dataset.")
            return df
        
        print(f"\nHandling missing values...")
        print(f"Missing values before imputation:\n{missing[missing > 0]}")
        
        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            if self.imputer_num is None:
                self.imputer_num = SimpleImputer(strategy='median')
                df[numerical_cols] = self.imputer_num.fit_transform(df[numerical_cols])
            else:
                df[numerical_cols] = self.imputer_num.transform(df[numerical_cols])
        
        # Impute categorical columns with most frequent
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            if self.imputer_cat is None:
                self.imputer_cat = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.imputer_cat.fit_transform(df[categorical_cols])
            else:
                df[categorical_cols] = self.imputer_cat.transform(df[categorical_cols])
        
        # Verify no missing values remain
        missing_after = df.isnull().sum()
        if missing_after.sum() > 0:
            print(f"Warning: Some missing values remain:\n{missing_after[missing_after > 0]}")
        else:
            print("All missing values have been imputed successfully.")
        
        return df
    
    def remove_irrelevant_features(self, df: pd.DataFrame, 
                                   target_col: str = 'isFraud') -> pd.DataFrame:
        """
        Remove irrelevant features and create balance difference features.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column to preserve
            
        Returns:
            DataFrame with irrelevant features removed and new features created
        """
        df = df.copy()
        
        print("\nRemoving irrelevant features and creating derived features...")
        
        # Drop high cardinality features and isFlaggedFraud
        columns_to_drop = ['nameDest', 'nameOrig', 'isFlaggedFraud']
        existing_drops = [col for col in columns_to_drop if col in df.columns]
        if existing_drops:
            df.drop(columns=existing_drops, inplace=True)
            print(f"  Dropped columns: {existing_drops}")
        
        # Create balance difference features
        if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
            df['balance_diff_org'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            print("  Created feature: balance_diff_org")
        
        if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
            df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            print("  Created feature: balance_diff_dest")
        
        # Drop original balance columns (information captured in differences)
        balance_cols_to_drop = ['oldbalanceOrg', 'newbalanceOrig', 
                               'oldbalanceDest', 'newbalanceDest']
        existing_balance_drops = [col for col in balance_cols_to_drop if col in df.columns]
        if existing_balance_drops:
            df.drop(columns=existing_balance_drops, inplace=True)
            print(f"  Dropped balance columns: {existing_balance_drops}")
        
        print(f"  Remaining columns: {list(df.columns)}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          categorical_cols: Optional[List[str]] = None,
                          drop_first: bool = True) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode. If None, auto-detects.
            drop_first: Whether to drop the first category (default True)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Filter to columns that actually exist
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if not categorical_cols:
            print("\nNo categorical columns found for encoding.")
            return df
        
        print(f"\nApplying one-hot encoding to: {categorical_cols}")
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
        
        print(f"  Original columns: {len(df.columns)}")
        print(f"  After encoding: {len(df_encoded.columns)}")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, 
                      numerical_cols: Optional[List[str]] = None,
                      target_col: str = 'isFraud') -> pd.DataFrame:
        """
        Scale numerical features using MinMaxScaler.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns to scale. If None, auto-detects.
            target_col: Name of target column to exclude from scaling
            
        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()
        
        if numerical_cols is None:
            # Auto-detect numerical columns, excluding target
            all_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in all_numerical if col != target_col]
        
        # Filter to columns that actually exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if not numerical_cols:
            print("\nNo numerical columns found for scaling.")
            return df
        
        print(f"\nScaling numerical features: {numerical_cols}")
        
        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            print("  Fitted MinMaxScaler")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            print("  Applied pre-fitted MinMaxScaler")
        
        return df
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series,
                              sampling_strategy: str = 'minority',
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sampling_strategy: SMOTE sampling strategy (default 'minority')
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        print(f"\nHandling class imbalance using SMOTE...")
        print(f"  Original class distribution:")
        print(f"    {y.value_counts().to_dict()}")
        
        # Initialize SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        
        # Apply SMOTE
        X_res, y_res = smote.fit_resample(X, y)
        
        print(f"  After SMOTE class distribution:")
        print(f"    {pd.Series(y_res).value_counts().to_dict()}")
        
        return X_res, y_res
    
    def preprocess(self, df: pd.DataFrame, 
                  target_col: str = 'isFraud',
                  handle_outliers: bool = True,
                  handle_missing: bool = True,
                  remove_irrelevant: bool = True,
                  encode_categorical: bool = True,
                  scale_features: bool = True,
                  handle_imbalance: bool = True,
                  return_X_y: bool = True) -> Tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            handle_outliers: Whether to handle outliers (default True)
            handle_missing: Whether to handle missing values (default True)
            remove_irrelevant: Whether to remove irrelevant features (default True)
            encode_categorical: Whether to encode categorical variables (default True)
            scale_features: Whether to scale numerical features (default True)
            handle_imbalance: Whether to handle class imbalance (default True)
            return_X_y: If True, returns (X, y), else returns processed DataFrame
            
        Returns:
            Tuple depending on return_X_y flag
        """
        df = df.copy()
        
        print("=" * 70)
        print("FRAUD DETECTION - PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"Initial shape: {df.shape}")
        
        # Step 1: Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df)
        
        # Step 2: Handle outliers
        if handle_outliers:
            df = self.handle_outliers_quantile(df)
        
        # Step 3: Remove irrelevant features and create derived features
        if remove_irrelevant:
            df = self.remove_irrelevant_features(df, target_col=target_col)
        
        # Step 4: Encode categorical variables
        if encode_categorical:
            df = self.encode_categorical(df)
        
        # Step 5: Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature columns for later use
        self.feature_columns = list(X.columns)
        
        # Step 6: Scale features
        if scale_features:
            X = self.scale_features(X, target_col=None)  # No target in X
        
        # Step 7: Handle class imbalance
        if handle_imbalance:
            X, y = self.handle_class_imbalance(X, y)
        
        print(f"\nFinal shape: X={X.shape}, y={y.shape}")
        print("=" * 70)
        
        if return_X_y:
            return X, y
        else:
            df_processed = X.copy()
            df_processed[target_col] = y
            return df_processed