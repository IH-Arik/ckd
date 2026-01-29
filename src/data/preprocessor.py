"""
Data Preprocessor for CKD Prediction
=====================================

Comprehensive data preprocessing pipeline for CKD clinical data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib


class DataPreprocessor:
    """
    Comprehensive data preprocessor for CKD prediction.
    
    Handles:
    - Missing value imputation
    - Feature scaling and encoding
    - Data validation
    - Train/test splitting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = None
        self.numeric_features = None
        self.categorical_features = None
        self.is_fitted = False
        
    def load_data(self, filepath: str, target: Optional[str] = 'ckd_status') -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            target: Target column name (None for no target)
            
        Returns:
            Features and target if target specified, otherwise just features
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape}")
            
            if target and target in data.columns:
                X = data.drop(columns=[target])
                y = data[target]
                self.target_column = target
                return X, y
            else:
                return data
                
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {str(e)}")
    
    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Update based on config if provided
        if 'numeric_features' in self.config:
            numeric_features = self.config['numeric_features']
        if 'categorical_features' in self.config:
            categorical_features = self.config['categorical_features']
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        return numeric_features, categorical_features
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline.
        
        Returns:
            ColumnTransformer for preprocessing
        """
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=self.config.get('knn_neighbors', 5))),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return preprocessor
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Self for method chaining
        """
        print("Fitting data preprocessor...")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Identify feature types
        self.identify_feature_types(X)
        
        # Create preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline()
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        
        self.is_fitted = True
        print("Preprocessor fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Ensure same columns as training data
        if not all(col in X.columns for col in self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        X_transformed = self.preprocessor.transform(X[self.feature_columns])
        
        print(f"Data transformed: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform the data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        return self.fit(X).transform(X)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature dataframe
            y: Target series
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        feature_names = []
        
        # Get numeric feature names
        feature_names.extend(self.numeric_features)
        
        # Get categorical feature names (after one-hot encoding)
        if hasattr(self.preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
            cat_features = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_features)
        
        return feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save the preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
        """
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor = preprocessor_data['preprocessor']
        self.config = preprocessor_data['config']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        self.numeric_features = preprocessor_data['numeric_features']
        self.categorical_features = preprocessor_data['categorical_features']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")
    
    def get_data_summary(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Get data summary statistics.
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'shape': X.shape,
            'missing_values': X.isnull().sum().to_dict(),
            'numeric_features': len(self.numeric_features) if self.numeric_features else 0,
            'categorical_features': len(self.categorical_features) if self.categorical_features else 0
        }
        
        if y is not None:
            summary['target_distribution'] = y.value_counts().to_dict()
            summary['target_balance'] = (y.value_counts(normalize=True) * 100).round(2).to_dict()
        
        return summary
