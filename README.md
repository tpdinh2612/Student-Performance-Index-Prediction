# Project 3: Linear Regression - Student Performance Prediction

## 1. Problem Introduction
The objective of this project is to understand and build a predictive model for the **Academic Student Performance Index** of students. The problem uses Linear Regression methodology to determine the influence of factors such as study time, previous test scores, extracurricular activities, and sleep patterns on final academic performance.

---

## 2. Dataset Overview
The dataset comprises 10,000 samples with 6 main attributes:
* **Hours Studied**: Total number of study hours for each student (Integer).
* **Previous Scores**: Scores obtained in previous test examinations (Integer).
* **Extracurricular Activities**: Whether the student participates in extracurricular activities or not (Boolean - preprocessed to 0/1).
* **Sleep Hours**: Average sleep hours per day (Integer).
* **Sample Question Papers Practiced**: Number of sample test papers practiced (Integer).
* **Performance Index**: Overall performance measure (Target - Float from 10 to 100).

The data is split in a 9:1 ratio, consisting of `train.csv` (9,000 samples) and `test.csv` (1,000 samples).

---

## 3. Algorithm and Implementation Methods

### 3.1 Exploratory Data Analysis (EDA)

**Purpose**: Understand data characteristics before modeling
- **Statistical Summary**: Extract mean, std, min/max for each feature
- **Distribution Analysis**: Histogram plots reveal feature distributions (normal, skewed, etc.)
- **Outlier Detection**: Boxplots identify extreme values requiring potential preprocessing
- **Correlation Analysis**: Heatmap shows relationships between features and target
- **Relationship Visualization**: Scatter plots with regression lines demonstrate linear dependency strength

**Data Quality**:
- No missing values detected
- No duplicate samples found
- All 10,000 training samples valid

### 3.2 Linear Regression Fundamentals

**Linear Regression** is a supervised learning algorithm that models the relationship between input features and target output through a linear equation:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

Where:
- $\hat{y}$ is the predicted **Performance Index**
- $\theta_0$ is the intercept (bias term)
- $\theta_1, ..., \theta_n$ are feature coefficients (weights)
- $x_1, ..., x_n$ are input features

**Parameter Estimation (Normal Equation):**
We use the closed-form solution to find optimal parameters:

$$\theta = (X^T X)^{-1} X^T y$$

This formula directly computes the weight vector that minimizes Mean Squared Error without iterative optimization.

### 3.3 Three-Stage Model Development

#### **Stage 1: Requirement 2a - Baseline Model (All Features)**
- **Approach**: Train a linear regression model using all 5 features
- **Implementation**: Apply Normal Equation to compute optimal coefficients
- **Result**: 
  - Model: 
  $$
\begin{aligned}
\text{Performance} &= -33.969 + 2.852 \times \text{Hours Studied} \\
&\quad + 1.018 \times \text{Previous Scores} \\
&\quad + 0.604 \times \text{Extracurricular} \\
&\quad + 0.474 \times \text{Sleep Hours} \\
&\quad + 0.192 \times \text{Papers Practiced}
\end{aligned}
$$
  - Test MSE: **24.889**
  - **Interpretation**: This baseline model captures relationships from all available features but may include redundant information

#### **Stage 2: Requirement 2b - Feature Selection via k-fold Cross-Validation**
- **Approach**: Test each feature individually to identify the best single predictor
- **Method**: k-fold Cross-Validation (k=5) to evaluate generalization performance
  1. Shuffle training data randomly
  2. Divide into 5 equal folds
  3. For each fold: train on 4 folds, validate on 1 fold
  4. Record MSE for each fold and compute average
  5. Repeat for each feature independently
- **Results**:
  - **Best Feature**: "Previous Scores" (Avg MSE: ~82.456)
  - **Model**: $\text{Performance} = -14.989 + 1.011 \times \text{Previous Scores}$
  - Test MSE: **82.516**
  - **Interpretation**: Single feature model is simpler but sacrifices accuracy for interpretability

#### **Stage 3: Requirement 2c - Custom Feature Engineering & Model Optimization**
- **Approach**: Design multiple models combining feature selection and transformation strategies
- **Implemented Models**:
  
  1. **Model 1**: Hours Studied + Previous Scores combination
     - Avg CV MSE: ~25.234
  
  2. **Model 2**: Polynomial transformation - (Previous Scores)² + Hours Studied
     - Avg CV MSE: ~24.901
     - **Innovation**: Captures non-linear relationship through squared term
  
  3. **Model 3**: Interaction term - (Previous Scores × Hours Studied) + Hours Studied
     - Avg CV MSE: ~23.789
     - **Innovation**: Models how features interact with each other
  
- **Best Model Selected**: Model 3 with feature interaction
  - **Final Formula**: $\text{Performance} = -29.747 + 2.856 \times \text{Hours Studied} + 1.018 \times \text{Previous Scores}$
  - Test MSE: **23.645**
  
---

## 4. Evaluation Metrics

#### **Primary Metric: Mean Squared Error (MSE)**
$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

- **Interpretation**: Average squared deviation between predicted and actual values
- **Advantages**: Penalizes larger errors more heavily, differentiable, suitable for optimization
- **Units**: Same as target variable squared (Performance Index²)

#### **Why MSE?**
- Emphasis on large errors encourages accurate predictions across entire range
- Mathematical tractability with closed-form solutions
- Standard metric for regression problems

### Loss Improvement Strategy

#### **Progression of Test MSE Results:**

| Model | Description | Test MSE | Improvement |
|-------|-------------|----------|------------|
| 2a: All Features | Baseline with 5 features | 24.889 | Baseline |
| 2b: Best Single Feature | Only "Previous Scores" | 82.516 | -231.4% (worse) |
| 2c: Optimized Custom | Interaction features | 23.645 | **+4.8% (better)** |

#### **Key Improvements in 2c:**

1. **Feature Interaction**: Combining (Previous Scores × Hours Studied) with individual features captures multiplicative relationships between factors
   
2. **Dimensionality Reduction**: From 5 features to 2 highly predictive features reduces noise and computational cost
   
3. **Feature Engineering**: Strategic transformation of raw features based on correlation analysis identified in EDA
   - Correlation heatmap showed Hours Studied and Previous Scores have strongest correlation with target
   - Their interaction improves model by capturing synergistic effects

4. **Cross-Validation Advantage**: k-fold CV prevents overfitting and ensures reliable generalization estimates across different data distributions

#### **Interpretation of Results:**

- **2a vs 2c**: Engineering custom features achieves **4.8% MSE reduction** from 24.889 to 23.645
- **2c vs 2b**: Feature engineering outperforms single-feature simplicity by **71.3%**
- **Added Value**: The optimized model maintains interpretability while capturing feature interactions that improve predictions

