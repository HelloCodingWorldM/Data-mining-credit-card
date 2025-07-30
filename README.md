# Credit Card Fraud Detection System

A comprehensive machine learning pipeline for detecting fraudulent credit card transactions using multiple algorithms including Logistic Regression, Random Forest, XGBoost, and Autoencoder-based anomaly detection.

## Overview

This project implements a complete end-to-end solution for credit card fraud detection, addressing the challenges of highly imbalanced datasets and the need for high precision in fraud identification. The system includes data preprocessing, multiple model implementations, performance evaluation, and business impact analysis.

## Features

- **Multiple ML Models**: Implements and compares various algorithms:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - Autoencoder (anomaly detection)
  
- **Imbalanced Data Handling**: 
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Random Undersampling
  - Class weight balancing
  
- **Comprehensive Evaluation**:
  - ROC-AUC, Precision-Recall curves
  - Matthew's Correlation Coefficient (MCC)
  - Business impact metrics (cost-benefit analysis)
  
- **Feature Analysis**:
  - SHAP (SHapley Additive exPlanations) for feature importance
  - Error analysis for false positives/negatives
  
- **Visualizations**:
  - Interactive plots using Plotly
  - Model comparison dashboards
  - Learning curves and performance metrics

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
tensorflow
shap
plotly
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place it in the project root directory

## Usage

### Basic Usage

```python
from fraud_detection import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline('creditcard.csv')

# Run complete pipeline
pipeline, results = main()
```

### Step-by-Step Execution

```python
# Load and explore data
df = pipeline.load_and_explore_data()
pipeline.visualize_data_distributions()

# Prepare data
pipeline.prepare_data()

# Train individual models
pipeline.train_logistic_regression()
pipeline.train_random_forest()
pipeline.train_xgboost()
pipeline.train_autoencoder()

# Analyze results
pipeline.plot_model_comparison()
pipeline.analyze_feature_importance()
pipeline.generate_report()
```

## Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py      # Main implementation file
├── requirements.txt        # Package dependencies
├── README.md              # This file
└── creditcard.csv         # Dataset (not included, download separately)
```

## Model Performance

The pipeline evaluates models on multiple metrics:

- **ROC-AUC Score**: Measures overall classification performance
- **Precision-Recall AUC**: Critical for imbalanced datasets
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Balanced metric for binary classification
- **Business Metrics**: Fraud prevention value vs. false positive costs

## Key Components

### FraudDetectionPipeline Class

The main class that orchestrates the entire workflow:
- Data loading and exploration
- Feature engineering and scaling
- Model training with hyperparameter tuning
- Performance evaluation and visualization
- Business impact analysis

### AdvancedFraudDetector Class

Additional utilities for:
- Creating interaction features
- Time-based feature engineering
- Ensemble prediction methods

## Results Interpretation

The pipeline generates:
1. **Model Comparison Report**: Performance metrics for all models
2. **Feature Importance Analysis**: SHAP values showing which features drive predictions
3. **Error Analysis**: Characteristics of false positives and false negatives
4. **Business Impact**: Estimated financial benefit of fraud prevention


## License

This project is part of an academic course. Please refer to your institution's guidelines for usage and distribution.

## Acknowledgments

- Dataset provided by [Machine Learning Group - ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Built using various open-source libraries in the Python ecosystem
