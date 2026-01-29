#!/usr/bin/env python3
"""
CKD Prediction System - Main Entry Point
========================================

Command-line interface for the CKD prediction system.

Usage:
    python src/main.py train --data data.csv --config config.yaml
    python src/main.py predict --model model.pkl --input input.csv
    python src/main.py evaluate --model model.pkl --test-data test.csv
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ensemble import CKDEnsemble
from src.data.preprocessor import DataPreprocessor
from src.utils.config import load_config
from src.utils.logger import setup_logger


def train_model(args):
    """Train the CKD prediction model."""
    logger = setup_logger("train")
    logger.info("Starting model training...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config.get('preprocessing', {}))
    
    # Load and preprocess data
    X, y = preprocessor.load_data(args.data)
    X_processed = preprocessor.fit_transform(X)
    
    # Initialize and train model
    model = CKDEnsemble(config.get('model', {}))
    model.train(X_processed, y)
    
    # Save model
    model.save(args.output)
    preprocessor.save(args.output.replace('.pkl', '_preprocessor.pkl'))
    
    logger.info(f"Model saved to {args.output}")


def predict(args):
    """Make predictions using trained model."""
    logger = setup_logger("predict")
    logger.info("Making predictions...")
    
    # Load model and preprocessor
    model = CKDEnsemble()
    model.load(args.model)
    
    preprocessor = DataPreprocessor()
    preprocessor.load(args.model.replace('.pkl', '_preprocessor.pkl'))
    
    # Load and preprocess data
    X = preprocessor.load_data(args.input, target=False)
    X_processed = preprocessor.transform(X)
    
    # Make predictions
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)
    
    # Save results
    results = X.copy()
    results['prediction'] = predictions
    results['probability'] = probabilities[:, 1]
    results.to_csv(args.output, index=False)
    
    logger.info(f"Predictions saved to {args.output}")


def evaluate(args):
    """Evaluate model performance."""
    logger = setup_logger("evaluate")
    logger.info("Evaluating model...")
    
    # Load model and preprocessor
    model = CKDEnsemble()
    model.load(args.model)
    
    preprocessor = DataPreprocessor()
    preprocessor.load(args.model.replace('.pkl', '_preprocessor.pkl'))
    
    # Load and preprocess test data
    X, y = preprocessor.load_data(args.test_data)
    X_processed = preprocessor.transform(X)
    
    # Evaluate
    metrics = model.evaluate(X_processed, y)
    
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CKD Prediction System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', required=True, help='Training data file')
    train_parser.add_argument('--config', help='Configuration file')
    train_parser.add_argument('--output', default='models/ckd_model.pkl', help='Output model file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Trained model file')
    predict_parser.add_argument('--input', required=True, help='Input data file')
    predict_parser.add_argument('--output', default='predictions.csv', help='Output predictions file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('--model', required=True, help='Trained model file')
    evaluate_parser.add_argument('--test-data', required=True, help='Test data file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
