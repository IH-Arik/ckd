"""
Ensemble Model for CKD Prediction
=================================

Advanced ensemble model combining multiple machine learning algorithms
for robust CKD prediction and prognosis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .base import BaseModel
from typing import Dict, Any, List, Tuple


class CKDEnsemble(BaseModel):
    """
    Comprehensive ensemble model for CKD prediction.
    
    Combines multiple algorithms including:
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Neural Networks
    - Logistic Regression
    - K-Nearest Neighbors
    - Naive Bayes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.ensemble_model = None
        self.base_models = {}
        self.scaler = StandardScaler()
        self.smote = BorderlineSMOTE(random_state=42)
        
    def _create_base_models(self) -> Dict[str, Any]:
        """Create base models for the ensemble."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=self.config.get('n_estimators', 100),
                depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42,
                verbose=False
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=self.config.get('hidden_layers', (100, 50)),
                max_iter=self.config.get('max_iter', 500),
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=self.config.get('max_iter', 500)
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=self.config.get('n_neighbors', 5),
                n_jobs=-1
            ),
            'naive_bayes': GaussianNB()
        }
        
        return models
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print("Training CKD Ensemble Model...")
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Store feature names and classes
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        # Create base models
        self.base_models = self._create_base_models()
        
        # Create pipelines for each model with preprocessing
        pipelines = {}
        for name, model in self.base_models.items():
            if name in ['neural_network', 'logistic_regression', 'knn']:
                # Models that benefit from scaling
                pipeline = ImbPipeline([
                    ('smote', self.smote),
                    ('scaler', self.scaler),
                    ('model', model)
                ])
            else:
                # Tree-based models
                pipeline = ImbPipeline([
                    ('smote', self.smote),
                    ('model', model)
                ])
            pipelines[name] = pipeline
        
        # Train individual models and collect performance
        model_scores = {}
        print("\nTraining individual models...")
        
        for name, pipeline in pipelines.items():
            print(f"Training {name}...")
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
            model_scores[name] = np.mean(cv_scores)
            print(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Fit the model on full data
            pipeline.fit(X, y)
            self.base_models[name] = pipeline
        
        # Select top performing models for ensemble
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 models for ensemble: {[model[0] for model in top_models]}")
        
        # Create voting ensemble with top models
        ensemble_estimators = [(name, self.base_models[name]) for name, _ in top_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        print("\nTraining ensemble model...")
        self.ensemble_model.fit(X, y)
        
        self.model = self.ensemble_model
        self.is_trained = True
        
        print("Ensemble training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get ensemble feature importance.
        
        Returns:
            Average feature importance across ensemble models
        """
        if not self.is_trained:
            return None
        
        importances = []
        
        for name, model in self.base_models.items():
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances.append(model.named_steps['model'].feature_importances_)
        
        if importances:
            return np.mean(importances, axis=0)
        
        return None
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get individual model performance scores.
        
        Returns:
            Dictionary of model names and their CV scores
        """
        if not self.is_trained:
            return {}
        
        performance = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'cv_score_'):
                performance[name] = model.cv_score_
        
        return performance
