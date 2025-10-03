# AI-ML Internship Portfolio - Elevate Labs

## Overview
This repository contains all projects and tasks completed during my AI & ML internship at Elevate Labs.

## Projects

### Task 1: Retail Product Data Cleaning and Preprocessing
- **Description**: Data cleaning pipeline for the Retail Product dataset featuring missing value handling, encoding, normalization, outlier visualization, and removal.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn
- **Status**: Completed ✅
- [View Project](./task1_cleaning/README.md)

### Task 2: Exploratory Data Analysis (EDA) - Housing Prices
- **Description**: Comprehensive exploratory data analysis on housing prices dataset featuring statistical analysis, correlation matrices, distribution visualizations, outlier detection, and pattern recognition.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn
- **Status**: Completed ✅
- [View Project](./task2-eda/README.md)
- **Key Features**:
  - Summary statistics and descriptive analysis
  - Correlation heatmaps and pairplots
  - Histograms and boxplots for distribution analysis
  - Interactive visualizations with Plotly
  - Outlier detection and anomaly analysis
  - Feature relationship insights

### Task 3: Linear Regression - Diabetes Disease Progression Prediction
- **Description**: Advanced linear regression analysis to predict diabetes disease progression using clinical features. Includes simple & multiple regression, regularization techniques, cross-validation, and comprehensive model evaluation.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy
- **Status**: Completed ✅
- [View Project](./task3-linear_regression/README.md)
- **Key Features**:
  - Simple vs Multiple Linear Regression comparison
  - Lasso and Ridge regularization implementation
  - 5-fold cross-validation for robust evaluation
  - Confidence intervals and residual analysis
  - Feature importance and coefficient interpretation
  - Comprehensive medical insights
- **Performance**:
  - R² Score: 0.4526 (45.3% variance explained)
  - MAE: 43.48 disease progression units
  - 94% improvement over simple regression model

### Task 4: Classification with Logistic Regression - Breast Cancer Diagnosis
- **Description**: Binary classification system using logistic regression to predict breast cancer diagnosis (malignant vs benign) with comprehensive model evaluation, threshold optimization, and sigmoid function analysis.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Status**: Completed ✅
- [View Project](./task4-logistic_regression2/README.md)
- **Key Features**:
  - End-to-end binary classification pipeline
  - ROC-AUC analysis and precision-recall curves
  - Threshold optimization for clinical decision making
  - Sigmoid function visualization and interpretation
  - Feature importance analysis with medical insights
  - Comprehensive model evaluation metrics
- **Performance**:
  - Accuracy: 98.83%
  - Precision: 99.07%
  - Recall: 99.07%
  - ROC-AUC: 99.81%
  - F1-Score: 99.07%

### Task 5: Car Evaluation Classification using Decision Trees and Random Forests
- **Description**: Advanced classification system for car acceptability prediction featuring Decision Trees, Random Forests, hyperparameter tuning with GridSearchCV, feature importance analysis, and comprehensive model evaluation.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Status**: Completed ✅
- [View Project](./task5-decision_tree/README.md)
- **Key Features**:
  - Decision Tree optimization with overfitting analysis
  - Random Forest ensemble implementation
  - Hyperparameter tuning with GridSearchCV (1080 model fits)
  - Feature importance analysis across multiple models
  - Cross-validation reliability testing
  - Comprehensive business insights and recommendations
- **Performance**:
  - Decision Tree (Optimized): 98.84% accuracy
  - Random Forest (Basic): 98.27% accuracy
  - Random Forest (Tuned): 98.27% accuracy
  - All models achieve >98% classification accuracy
- **Key Insights**:
  - Safety is the most critical factor (27.5% importance)
  - Capacity is second most important (22.6%)
  - Ensemble methods provide better generalization
  - Cross-validation shows Random Forest as most reliable

### Task 6: K-Nearest Neighbors (KNN) Classification - Handwritten Digit Recognition
- **Description**: Comprehensive handwritten digit classification system using K-Nearest Neighbors algorithm with advanced hyperparameter optimization, distance metric analysis, weighting strategy comparison, and comprehensive model evaluation achieving 98.15% accuracy.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Status**: Completed ✅
- [View Project](./task6-KNN/README.md)
- **Key Features**:
  - Systematic K value optimization (1-20) with cross-validation
  - Distance metric comparison (Euclidean, Manhattan, Chebyshev, Minkowski)
  - Weighting strategy analysis (uniform vs distance-based)
  - Decision boundary visualization using PCA
  - Comprehensive error analysis and misclassification patterns
  - Model comparison against SVM and Random Forest
- **Performance**:
  - **KNN (Optimized)**: 98.15% accuracy
  - **Best Parameters**: K=1, Manhattan distance, distance weighting
  - **Cross-validation**: 97.21% accuracy
  - **Weighting Improvement**: +0.40% accuracy gain
- **Key Insights**:
  - K=1 works best indicating well-separated digit classes
  - Manhattan distance outperforms Euclidean for image data
  - Distance weighting provides consistent improvements across most K values
  - Model achieves competitive performance against complex algorithms
  - Excellent computational efficiency (0.002s training time)

### Task 7: Support Vector Machines (SVM) - Breast Cancer Diagnosis
- **Description**: Advanced breast cancer diagnosis system using Support Vector Machines with comprehensive feature analysis, kernel optimization, hyperparameter tuning, and clinical performance evaluation achieving 98.2% accuracy.
- **Technologies**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Status**: Completed ✅
- [View Project](./task7-SVM/README.md)
- **Key Features**:
  - Comprehensive feature analysis and clinical interpretation
  - Kernel function comparison (Linear, RBF, Polynomial, Sigmoid)
  - Hyperparameter optimization (C, gamma) with grid search
  - Feature importance analysis with medical insights
  - Clinical performance metrics (sensitivity, specificity, ROC-AUC)
  - Decision boundary visualization using PCA
  - Model comparison against other classifiers
- **Performance**:
  - **Accuracy**: 98.2%
  - **Sensitivity**: 97.1% (crucial for cancer detection)
  - **Specificity**: 98.9%
  - **ROC-AUC**: 99.3%
  - **Best Parameters**: RBF kernel, C=1.0, gamma='scale'
- **Key Insights**:
  - Worst concave points and perimeter are strongest cancer indicators
  - RBF kernel outperforms others for non-linear medical data
  - Moderate regularization (C=1) provides optimal bias-variance tradeoff
  - Model achieves performance comparable to expert radiologists
  - Suitable for clinical decision support applications

## Progress Tracking

| Task | Project Name                                       | Status    | Completion Date |
|------|----------------------------------------------------|-----------|-----------------|
| 1    | Retail Product Data Cleaning and Preprocessing     | Completed | 2025-09-22      |
| 2    | Exploratory Data Analysis (EDA) - Housing Prices   | Completed | 2025-09-23      |
| 3    | Linear Regression - Diabetes Disease Progression   | Completed | 2025-09-24      |
| 4    | Logistic Regression - Breast Cancer Diagnosis      | Completed | 2025-09-25      |
| 5    | Decision Trees & Random Forests - Car Evaluation   | Completed | 2025-09-29      |
| 6    | KNN Classification - Handwritten Digit Recognition | Completed | 2025-09-30      |
| 7    | SVM Classification - Breast Cancer Diagnosis       | Completed | 2025-10-01      |

## Installation
Each project has its own dependencies. Please refer to the respective project README for environment setup and running instructions.

## Skills Demonstrated

### Technical Skills
- **Data Preprocessing**: Missing value handling, outlier detection, normalization, encoding
- **Exploratory Data Analysis**: Statistical analysis, correlation studies, visualization
- **Machine Learning**: Linear regression, logistic regression, regularization, model evaluation, cross-validation
- **Classification Algorithms**: Binary classification, threshold optimization, ROC analysis, Decision Trees, Random Forests, K-Nearest Neighbors, Support Vector Machines
- **Ensemble Methods**: Random Forest implementation, hyperparameter tuning, feature importance
- **Distance-Based Algorithms**: KNN optimization, distance metric analysis, weighting strategies
- **Kernel Methods**: SVM with multiple kernel functions, parameter optimization
- **Advanced Optimization**: GridSearchCV, automated parameter selection, model comparison
- **Data Visualization**: Matplotlib, Seaborn, Plotly, correlation heatmaps, residual plots, ROC curves, decision trees, confusion matrices, decision boundaries
- **Python Libraries**: Pandas, NumPy, Scikit-learn, SciPy, pmdarima

### Analytical Skills
- Statistical analysis and interpretation
- Model performance evaluation and comparison
- Feature importance analysis across multiple algorithms
- Medical/domain insights extraction (healthcare diagnostics, automotive evaluation, image recognition)
- Precision-recall tradeoff analysis
- Overfitting detection and prevention
- Ensemble method reliability assessment
- Distance metric impact analysis
- Kernel selection and parameter optimization
- Hyperparameter sensitivity analysis
- Data-driven decision making
- Clinical performance evaluation (sensitivity, specificity)

## Learning Outcomes

Through these projects, I have developed strong competencies in:
- End-to-end data science pipeline implementation
- Advanced statistical analysis and visualization
- Machine learning model development and evaluation (regression, classification, ensemble methods, distance-based algorithms, kernel methods)
- Domain-specific insight extraction (retail, real estate, healthcare diagnostics, automotive, image recognition)
- Medical analytics and clinical decision support systems
- Hyperparameter optimization and model tuning
- Ensemble method implementation and evaluation
- Distance-based algorithm optimization and analysis
- SVM kernel selection and parameter optimization
- Professional documentation and code organization
- Business intelligence extraction from machine learning models

## Project Evolution

The internship projects demonstrate progressive complexity:
1. **Foundation**: Data cleaning and preprocessing fundamentals
2. **Analysis**: Exploratory data analysis and visualization techniques
3. **Regression**: Linear models with regularization and validation
4. **Classification**: Binary classification with medical applications
5. **Advanced ML**: Ensemble methods, tree-based algorithms, hyperparameter tuning
6. **Distance-Based ML**: KNN optimization, distance metrics, weighting strategies for image recognition
7. **Kernel Methods**: SVM with kernel optimization for complex decision boundaries

## Algorithm Portfolio Coverage

| Algorithm Category          | Projects | Key Techniques |
|-----------------------------|----------|----------------|
| **Regression**              | Task 3   | Linear Regression, Regularization, Cross-validation |
| **Binary Classification**   | Task 4   | Logistic Regression, ROC Analysis, Threshold Optimization |
| **Ensemble Methods**        | Task 5   | Decision Trees, Random Forests, GridSearchCV |
| **Distance-Based**          | Task 6   | KNN, Distance Metrics, Weighting Strategies |
| **Kernel Methods**          | Task 7   | SVM, Kernel Selection, Hyperparameter Tuning |
| **Multi-class Classification** | Task 6   | Handwritten Digit Recognition, Confusion Matrices |
| **Medical Diagnostics**     | Task 4, 7 | Clinical Performance Metrics, Sensitivity Analysis |

## Clinical Applications Expertise

### Healthcare Diagnostics
- **Breast Cancer Diagnosis** (Tasks 4 & 7): Developed multiple classification systems achieving >98% accuracy
- **Diabetes Progression Prediction** (Task 3): Regression analysis for disease monitoring
- **Clinical Feature Interpretation**: Medical insights from feature importance analysis
- **Safety-Critical Performance**: Focus on sensitivity and false negative minimization

### Technical Competencies in Medical AI
- ROC-AUC analysis for diagnostic performance
- Sensitivity-specificity tradeoff optimization
- Clinical decision threshold selection
- Feature importance with medical relevance
- Model interpretability for healthcare applications
- Ethical considerations in medical AI

## Contact
- **Name:** Ghanashyam T V  
- **Email:** [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)  
- **LinkedIn:** [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)  

---

*Elevate Labs AI & ML Internship Portfolio - Demonstrating comprehensive machine learning expertise across regression, classification, ensemble methods, distance-based algorithms, and kernel methods with specialized focus on healthcare diagnostics*