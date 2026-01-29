"""
Base Model Class
================

Abstract base class for all CKD prediction models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, Any, Tuple, Optional


class BaseModel(ABC):
    """Abstract base class for CKD prediction models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.classes_ = None
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test feature matrix
            y: Test target vector
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba[:, 1]),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'classes_': self.classes_
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.classes_ = model_data['classes_']
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def set_feature_names(self, feature_names: list) -> None:
        """
        Set feature names.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
