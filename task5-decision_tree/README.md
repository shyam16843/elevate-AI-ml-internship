# Car Evaluation Classification using Decision Trees and Random Forests

## Project Description

This project performs comprehensive car acceptability classification using machine learning algorithms including Decision Trees and Random Forests. It features automated hyperparameter tuning, detailed feature importance analysis, cross-validation reliability testing, and comprehensive model evaluation. Visualizations include decision tree structures, feature importance comparisons, confusion matrices, and performance metrics for actionable business insights.

---

## 1. Project Objective

Develop a reliable and interpretable classification system for car evaluation that can:

- Analyze and preprocess categorical car features with meaningful ordering
- Build and optimize Decision Tree classifiers with overfitting analysis
- Implement ensemble Random Forest models with hyperparameter tuning
- Compare model performance using accuracy, F1-score, and cross-validation
- Identify key factors influencing car acceptability through feature importance analysis
- Generate strategic business recommendations based on model insights

---

## 2. Dataset Information

- **Source**: UCI Machine Learning Repository - Car Evaluation Dataset
- **Records**: 1,728 car evaluations
- **Features**: 6 categorical attributes
- **Target**: 4 acceptability classes
- **Attributes**: 
  - Buying price (low, med, high, vhigh)
  - Maintenance cost (low, med, high, vhigh)
  - Number of doors (2, 3, 4, 5more)
  - Capacity (2, 4, more)
  - Luggage boot size (small, med, big)
  - Safety (low, med, high)
- **Target Classes**: unacceptable, acceptable, good, very_good

---

## 3. Methodology

### Data Preparation and Preprocessing

- Loads car evaluation data from UCI repository with proper column naming
- Converts categorical features to ordered categories with meaningful sequencing
- Applies Label Encoding while preserving feature relationships
- Splits data into training (80%) and testing (20%) sets with stratification

### Exploratory Data Analysis

- Analyzes feature distributions and class balances
- Provides descriptive statistics of the encoded dataset
- Visualizes data structure and relationships

### Model Building & Optimization

#### Decision Tree Analysis
- Trains initial Decision Tree classifier with default parameters
- Performs overfitting analysis by testing tree depths (1-15)
- Identifies optimal tree depth through training vs test accuracy comparison
- Visualizes optimized decision tree structure and rules

#### Random Forest Implementation
- Builds basic Random Forest with 100 estimators
- Implements advanced hyperparameter tuning using GridSearchCV
- Tests 216 parameter combinations with 5-fold cross-validation
- Selects best model based on cross-validation performance

### Model Evaluation & Comparison

- Compares all four models: DT Initial, DT Optimized, RF Basic, RF Tuned
- Evaluates using accuracy, precision, recall, F1-score metrics
- Performs comprehensive cross-validation reliability analysis
- Generates confusion matrices for detailed error analysis

### Feature Importance Analysis

- Extracts and compares feature importances across all models
- Identifies most influential factors in car acceptability decisions
- Visualizes importance rankings for business insights

---

## 4. Key Features Implemented

### Core Functionality

- Automated hyperparameter tuning with GridSearchCV (1080 model fits)
- Overfitting detection and prevention through optimal depth selection
- Comprehensive model comparison across multiple metrics
- Feature importance analysis with cross-model comparison
- Business-translatable insights and recommendations

### Technical Features

- Proper categorical data encoding with meaningful ordering
- Stratified train-test splitting for representative evaluation
- Cross-validation with reliability intervals
- Confusion matrix visualization and analysis
- Modular Python implementation using scikit-learn, pandas, and matplotlib

### Advanced ML Techniques

- Ensemble methods with Random Forests
- Hyperparameter optimization with exhaustive grid search
- Model interpretability through decision tree visualization
- Feature importance analysis for business intelligence

---

## 5. Project Setup and Requirements

### Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Installation

Install dependencies by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 6. Running the Project

1. The project automatically downloads the dataset from UCI repository
2. Run the main script:

```bash
python decision.py
```

### The system will automatically:

- Download and preprocess the car evaluation dataset
- Perform exploratory data analysis
- Train and optimize Decision Tree classifiers
- Implement and tune Random Forest models
- Generate comprehensive performance comparisons
- Produce visualizations and business insights
- Save analysis results and recommendations

---

## 7. Key Results

### Model Performance
- **Decision Tree (Initial)**: 98.55% accuracy
- **Decision Tree (Optimized)**: 98.84% accuracy (Best)
- **Random Forest (Basic)**: 98.27% accuracy
- **Random Forest (Tuned)**: 98.27% accuracy

### Feature Importance Ranking
1. **Safety** (27.5%) - Most critical factor
2. **Capacity** (22.6%) - Second most important
3. **Buying Price** (18.7%)
4. **Maintenance Cost** (15.8%)
5. **Luggage Boot** (8.5%)
6. **Doors** (7.0%) - Least important

### Cross-Validation Reliability
- Decision Tree: 77.8% ± 20.4%
- Random Forest (Basic): 81.6% ± 11.5% (Most Reliable)
- Random Forest (Tuned): 80.5% ± 15.0%

---

## 8. Visualization Overview
A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](Visualization.md). This document includes detailed descriptions and analyses of all key plots

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.

---

## 9. Business Insights & Recommendations

### Key Findings
- Safety is the dominant factor in car acceptability across all models
- All models achieve exceptional performance (>98% accuracy)
- Ensemble methods provide better generalization and reliability
- Hyperparameter tuning maintains high performance with optimized parameters

### Strategic Recommendations
- **Product Development**: Prioritize safety features in car design
- **Marketing**: Emphasize safety and capacity in advertising campaigns
- **Model Deployment**: Use Optimized Decision Tree for maximum accuracy
- **Monitoring**: Implement cross-validation for ongoing model validation
- **Feature Focus**: Allocate resources based on feature importance rankings

---

## 10. Technical Architecture

```
Data Acquisition → Preprocessing → Model Training → Hyperparameter Tuning
       ↓
Business Insights ← Model Evaluation ← Feature Analysis ← Cross-Validation
```

### Model Pipeline
1. **Data Layer**: UCI repository → pandas DataFrame
2. **Preprocessing**: Categorical encoding → train-test split
3. **Model Layer**: Decision Trees → Random Forests → GridSearchCV
4. **Evaluation**: Accuracy metrics → confusion matrices → cross-validation
5. **Insight Layer**: Feature importance → business recommendations

---

## 11. Future Enhancements

### Technical Improvements
- Implement additional ensemble methods (Gradient Boosting, XGBoost)
- Add automated feature engineering and selection
- Develop web interface for interactive model exploration
- Implement model deployment API for real-time predictions

### Business Applications
- Extend to multi-class probability forecasting
- Develop recommendation system for car manufacturers
- Create dashboard for continuous model monitoring
- Integrate with automotive industry data sources

### Advanced Analytics
- Incorporate SHAP values for enhanced interpretability
- Implement automated model retraining pipelines
- Add anomaly detection for unusual car evaluations
- Develop A/B testing framework for model improvements

---

## 12. Contact

For questions, collaboration, or feedback:

- **Name**: Ghanashyam T V
- **Email**: [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

## 13. Acknowledgments

- **Data Source**: UCI Machine Learning Repository
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn
- **Methodology**: Based on industry-standard machine learning practices

---

Thank you for exploring the Car Evaluation Classification project! This demonstration showcases advanced machine learning skills with practical business applications in the automotive industry. The project provides a complete workflow from data acquisition to actionable business intelligence.