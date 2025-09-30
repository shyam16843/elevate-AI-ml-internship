# Handwritten Digit Recognition using K-Nearest Neighbors (KNN)

## Project Description

This project implements a comprehensive handwritten digit classification system using the K-Nearest Neighbors (KNN) algorithm. It features advanced hyperparameter optimization, distance metric analysis, weighting strategy comparison, and comprehensive model evaluation. Visualizations include digit samples, decision boundaries, confusion matrices, and performance comparisons. The system achieves 98.15% accuracy through systematic parameter tuning and provides detailed insights into KNN behavior for image classification tasks.

---

## 1. Project Objective

Develop an accurate and interpretable digit recognition system that can:

- Preprocess and normalize 8x8 pixel handwritten digit images
- Systematically optimize KNN hyperparameters (K value, distance metrics, weighting)
- Compare different distance metrics and weighting strategies
- Visualize decision boundaries and model predictions
- Analyze misclassifications and model confidence
- Provide comprehensive performance evaluation with cross-validation

---

## 2. Dataset Information

- **Source**: Scikit-learn Digits Dataset
- **Records**: 1,797 handwritten digit samples
- **Features**: 64 pixels (8x8 grayscale images)
- **Target**: 10 digit classes (0-9)
- **Pixel Values**: 0-16 (grayscale intensity)
- **Image Size**: 8x8 pixels
- **Classes**: Balanced distribution across all 10 digits

### Class Distribution:
- Digit 0: 178 samples
- Digit 1: 182 samples  
- Digit 2: 177 samples
- Digit 3: 183 samples
- Digit 4: 181 samples
- Digit 5: 182 samples
- Digit 6: 181 samples
- Digit 7: 179 samples
- Digit 8: 174 samples
- Digit 9: 180 samples

---

## 3. Methodology

### Data Preparation and Preprocessing

- Loads digits dataset from scikit-learn with 8x8 pixel images
- Performs train-test split (70-30) with stratification for balanced classes
- Applies StandardScaler for feature normalization (crucial for distance-based algorithms)
- Reshapes data for visualization and analysis

### Exploratory Data Analysis

- Visualizes sample handwritten digits with true labels
- Analyzes class distribution and dataset balance
- Examines pixel value statistics and data characteristics
- Provides dataset overview and basic statistics

### Hyperparameter Optimization

#### K Value Selection
- Tests K values from 1 to 20 using cross-validation
- Compares training, test, and cross-validation accuracy
- Identifies optimal K through error analysis and performance curves
- Analyzes bias-variance tradeoff across different K values

#### Distance Metric Analysis
- Compares four distance metrics: Euclidean, Manhattan, Chebyshev, Minkowski
- Evaluates each metric with cross-validation across all K values
- Selects best performing distance metric for final model
- Analyzes metric-specific optimal K values

#### Weighting Strategy Comparison
- Tests uniform vs distance-based weighting approaches
- Evaluates performance across K values 1-15
- Analyzes computational cost and accuracy tradeoffs
- Identifies optimal weighting strategy through systematic comparison

### Model Evaluation & Comparison

- Compares KNN performance against SVM and Random Forest
- Evaluates using accuracy, precision, recall, F1-score metrics
- Performs 5-fold cross-validation for reliability assessment
- Generates confusion matrices for detailed error analysis
- Analyzes prediction confidence and misclassification patterns

### Advanced Visualization

- Visualizes decision boundaries using PCA dimensionality reduction
- Displays correct and incorrect predictions with actual images
- Analyzes common misclassification patterns
- Creates comprehensive performance comparison charts

---

## 4. Key Features Implemented

### Core Functionality

- Systematic hyperparameter optimization (K value, distance metrics, weighting)
- Advanced distance metric comparison with performance analysis
- Weighting strategy evaluation (uniform vs distance-based)
- Comprehensive model validation with cross-validation
- Detailed error analysis and misclassification patterns

### Technical Features

- Proper feature scaling for distance-based algorithms
- Stratified sampling for representative evaluation
- PCA-based decision boundary visualization
- Computational performance profiling
- Modular Python implementation using scikit-learn, pandas, and matplotlib

### Advanced ML Techniques

- KNN with multiple distance metrics and weighting strategies
- Cross-validation with reliability assessment
- Model interpretability through decision boundary visualization
- Feature importance analysis through performance metrics
- Ensemble method comparison for contextual performance

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

1. Save the script as `knn_digits_classification.py`
2. Run the main script:

```bash
python knn_digits_classification.py
```

### The system will automatically:

- Load and preprocess the digits dataset
- Perform exploratory data analysis and visualization
- Optimize K value through systematic testing
- Compare distance metrics and select optimal one
- Evaluate weighting strategies (uniform vs distance)
- Train final optimized model
- Generate comprehensive performance evaluation
- Produce visualizations and technical insights
- Compare against other ML algorithms (SVM, Random Forest)

---

## 7. Key Results

### Optimal Parameters Found
- **Best K Value**: 1
- **Best Distance Metric**: Manhattan
- **Best Weighting Strategy**: Distance-based
- **Cross-validation Accuracy**: 97.21%

### Model Performance
- **KNN (Initial)**: 97.59% accuracy
- **KNN (Optimized)**: 98.15% accuracy (Best)
- **SVM**: 98.33% accuracy
- **Random Forest**: 96.67-97.22% accuracy

### Weighting Strategy Impact
- **Distance Weighting**: Improved accuracy by 0.40%
- **Performance Improvement**: 14 out of 15 K values showed improvement
- **Best Improvement**: +1.43% at K=2
- **Average Improvement**: +0.42% across all K values

### Computational Performance
- **Training Time**: 0.001-0.002 seconds
- **Prediction Time**: 0.022-0.025 seconds
- **Time per Prediction**: 0.000042-0.000046 seconds

---

## 8. Visualization Overview

A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](Visualization.md). This document includes detailed descriptions and analyses of all key plots that demonstrate the KNN algorithm's behavior and performance.

### Visualization Categories Included:

#### **Data Exploration Visualizations**
- **Sample Handwritten Digits**: 15 examples showing the raw 8x8 pixel images with true labels
- **Class Distribution Analysis**: Bar charts showing balanced distribution across all 10 digits with both linear and logarithmic scales

#### **Hyperparameter Optimization Visualizations**
- **K Value Analysis**: Three-panel plot showing accuracy vs K, error rate vs K, and zoomed optimal region
- **Distance Metrics Comparison**: Line plots comparing four distance metrics across K values with performance rankings
- **Weighting Strategies**: Four-panel comprehensive analysis of uniform vs distance weighting

#### **Model Performance Visualizations**
- **Confusion Matrices**: Standard and normalized versions showing detailed classification patterns
- **Prediction Samples**: Grids of correctly and incorrectly classified digits with true vs predicted labels
- **Decision Boundaries**: PCA-reduced 2D visualizations showing how KNN separates digit classes

#### **Technical Analysis Visualizations**
- **Performance Difference Analysis**: Bar charts showing when distance weighting helps vs hurts
- **Computational Cost**: Training time comparisons between different configurations
- **Small K Analysis**: Detailed focus on the range where weighting strategies matter most

### Accessing Visualizations

The actual plot images generated by the script are saved automatically when you run the code. For the complete visualization document:

1. **Run the main script** to generate all plots in real-time
2. **Review the [Visualization Document](Visualization.md)** for detailed analysis of each plot
3. **All plots are displayed interactively** during script execution and can be saved for documentation

### Key Visualization Insights

- **Optimal K Identification**: Clear visualization of the bias-variance tradeoff
- **Metric Performance**: Manhattan distance consistently outperforms other metrics
- **Weighting Impact**: Distance weighting provides measurable improvements across most K values
- **Error Patterns**: Specific digit confusions (8→1, 5→9) are clearly identified
- **Decision Boundaries**: Complex multi-class separation in reduced feature space

We recommend reviewing the visualization document alongside this README for a thorough understanding of the KNN model's performance, parameter sensitivities, and classification behavior. The visualizations provide intuitive insights into why certain parameters work best and how the model makes decisions.

---

## 9. Technical Insights

### Algorithm Behavior
- **K=1 worked best** indicating well-separated classes in feature space
- **Manhattan distance outperformed Euclidean** for this image dataset
- **Distance weighting provided consistent improvements** across most K values
- **Model shows excellent performance** despite simple algorithm

### Performance Characteristics
- **Fast training** (lazy learning - just stores data)
- **Slower prediction** (computes distances to all training samples)
- **Excellent for multi-class problems** (naturally handles 10 digits)
- **Sensitive to feature scaling** (normalization was crucial)

### Error Analysis
- **Most common misclassification**: 8→1 (4 instances)
- **Second most common**: 5→9 (3 instances)
- **Overall error rate**: 2.4% (13 misclassifications out of 540)
- **Class-wise performance**: All digits >90% accuracy

---

## 10. Business Insights & Recommendations

### Key Findings
- KNN achieves exceptional performance (98.15%) on handwritten digits
- Simple algorithms can compete with complex models for well-structured image data
- Parameter tuning significantly impacts KNN performance
- Distance-based weighting provides robust improvements

### Strategic Recommendations
- **Model Selection**: Use optimized KNN for interpretable, high-accuracy digit recognition
- **Parameter Tuning**: Always optimize K, distance metric, and weighting strategy
- **Preprocessing**: Feature scaling is essential for distance-based algorithms
- **Deployment**: Suitable for applications requiring fast training and moderate prediction speeds

### Application Scenarios
- **Postal Services**: Automated zip code reading
- **Banking**: Check amount digitization
- **Education**: Automated test grading
- **Mobile Applications**: Handwritten note recognition

---

## 11. Technical Architecture

```
Data Loading → Preprocessing → K Optimization → Metric Comparison
       ↓
Business Insights ← Model Evaluation ← Weighting Analysis ← Final Training
```

### Model Pipeline
1. **Data Layer**: Scikit-learn digits → numpy arrays
2. **Preprocessing**: StandardScaler normalization → train-test split
3. **Optimization Layer**: K selection → metric comparison → weighting strategy
4. **Evaluation**: Accuracy metrics → confusion matrices → cross-validation
5. **Insight Layer**: Error analysis → performance comparison → business applications

---

## 11. Future Enhancements

### Technical Improvements
- Implement K-D Trees for faster nearest neighbor searches
- Add advanced distance metrics (Cosine, Mahalanobis)
- Develop ensemble methods with multiple KNN models
- Implement automated hyperparameter tuning with Bayesian optimization

### Advanced Applications
- Extend to larger image datasets (MNIST, EMNIST)
- Develop real-time digit recognition system
- Create web interface for interactive digit classification
- Implement transfer learning for domain adaptation

### Performance Optimization
- Parallelize distance computations for faster predictions
- Implement approximate nearest neighbor algorithms
- Add GPU acceleration for large-scale deployments
- Develop model compression techniques

---


## 13. Contact

For questions, collaboration, or feedback:

- **Name**: Ghanashyam T V
- **Email**: [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)


---

## 14. Acknowledgments

- **Data Source**: Scikit-learn Digits Dataset
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, numpy
- **Methodology**: Based on industry-standard machine learning practices for KNN optimization

---

Thank you for exploring the Handwritten Digit Recognition using KNN project! This demonstration showcases comprehensive machine learning skills with practical applications in image classification. The project provides a complete workflow from data exploration to optimized model deployment with detailed technical insights and business recommendations.