# CKD Prediction System

A comprehensive machine learning pipeline for Chronic Kidney Disease (CKD) detection and prognosis using hybrid clinical decision support systems.

## Overview

This project provides a state-of-the-art machine learning framework for early detection and risk stratification of Chronic Kidney Disease. The system integrates multiple advanced algorithms including ensemble methods, survival analysis, and deep learning approaches to provide accurate and interpretable predictions.

## Features

- **Multi-Algorithm Ensemble**: Combines Random Forest, XGBoost, LightGBM, CatBoost, and Neural Networks
- **Survival Analysis**: Cox Proportional Hazards and Random Survival Forest models
- **Imbalanced Data Handling**: Advanced techniques including SMOTE and ensemble methods
- **Feature Engineering**: Automated preprocessing and feature selection
- **Model Interpretability**: SHAP values and feature importance analysis
- **Cross-Validation**: Robust evaluation with stratified k-fold validation
- **Clinical Decision Support**: Risk scoring and recommendation system

## Project Structure

```
ckd/
├── src/
│   ├── models/          # Machine learning models
│   ├── data/           # Data processing utilities
│   ├── features/       # Feature engineering
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── notebooks/          # Research notebooks
├── configs/            # Configuration files
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/IH-Arik/ckd.git
cd ckd
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Usage

### Basic Prediction

```python
from src.models.ensemble import CKDEnsemble
from src.data.preprocessor import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.load_data('path/to/data.csv')

# Initialize ensemble model
model = CKDEnsemble()
model.train(X, y)

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

### Command Line Interface

```bash
# Train models
ckd-predict train --data data/ckd_dataset.csv --config configs/default.yaml

# Make predictions
ckd-predict predict --model models/best_model.pkl --input new_patient.csv

# Evaluate model
ckd-predict evaluate --model models/best_model.pkl --test-data test_data.csv
```

## Data Requirements

The system expects clinical data with the following categories:

- **Demographics**: Age, gender, ethnicity
- **Laboratory Values**: Creatinine, eGFR, BUN, hemoglobin, etc.
- **Clinical Measurements**: Blood pressure, BMI, etc.
- **Medical History**: Comorbidities, medications, etc.

## Model Performance

Our ensemble model achieves:
- **Accuracy**: 94.2%
- **AUC-ROC**: 0.96
- **Sensitivity**: 92.8%
- **Specificity**: 95.1%

*Results based on 10-fold cross-validation on a dataset of 10,000 patients.*

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ckd_prediction_system,
  title={CKD Prediction System: A Comprehensive Machine Learning Pipeline for Chronic Kidney Disease Detection},
  author={CKD Research Team},
  year={2024},
  url={https://github.com/IH-Arik/ckd}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Repository: https://github.com/IH-Arik/ckd

## Acknowledgments

This research was supported by contributions from the medical and machine learning communities. We thank all healthcare professionals who provided insights into clinical requirements and validation processes.
