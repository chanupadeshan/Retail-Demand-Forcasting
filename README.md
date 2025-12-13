# A Methodological Blueprint for a High-Impact Comparative Analysis of Time-Series Forecasting Models

## 🔬 Advanced, Research-Level Description of the Methodological Blueprint

This document outlines a scholarly framework for a comparative analysis of time series forecasting models, specifically targeting a research-level contribution suitable for an MSc thesis or academic publication. It mandates a shift from a simplistic "which is better?" comparison to a nuanced exploration of how architectural principles dictate model behavior and performance.

The blueprint is structured around four non-negotiable pillars of methodological rigor:

---

## 1. Theoretical Foundations and Architectural Philosophies

The analysis is grounded in the fundamental philosophical differences between four models spanning the spectrum of modeling approaches:

### **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
- **Philosophy**: Rooted in parsimony and statistical rigor using the Box-Jenkins methodology
- **Core Characteristic**: Strictly dependent on the underlying time series being stationary (after differencing)
- **Parameter Identification**: Relies on ACF/PACF plots for parameter identification
- **Methodological Strength**: Transparent, interpretable, and theoretically grounded in classical statistical theory

### **Prophet**
- **Philosophy**: Embodies practicality and robustness, developed at Meta for production forecasting
- **Decomposition Structure**: Uses additive decomposition: $y(t) = g(t) + s(t) + h(t) + \epsilon(t)$
  - $g(t)$: Trend component
  - $s(t)$: Seasonality via Fourier series
  - $h(t)$: Holiday effects
  - $\epsilon(t)$: Error term
- **Key Advantage**: Explicitly eschews the strict stationarity requirement of SARIMA, handling trend breaks and irregular patterns

### **XGBoost (Extreme Gradient Boosting)**
- **Philosophy**: Represents a supervised learning paradigm that transforms the forecasting task into a regression problem
- **Critical Dependency**: Success is contingent upon creative and relevant feature engineering
- **Feature Engineering Requirements**: Lagged features, rolling statistics, time-based features, and domain-specific indicators
- **Methodological Innovation**: Enables the incorporation of exogenous variables naturally through the gradient boosting framework

### **Temporal Fusion Transformer (TFT)**
- **Philosophy**: State-of-the-art deep learning architecture designed to be both highly accurate and interpretable
- **Core Innovation**: Ability to handle heterogeneous input types:
  - Static features (non-time-varying attributes)
  - Known-future features (e.g., scheduled events, promotions)
  - Historical exogenous variables
- **Architectural Components**:
  - **Gated Residual Networks (GRNs)**: Enable adaptive feature selection and information flow
  - **Multi-Head Self-Attention**: Captures temporal dependencies and variable interactions
- **Interpretability Focus**: Provides variable importance scores and temporal attention weights

---

## 2. Rigorous Data Preprocessing and Validation Strategies

The blueprint emphasizes that the integrity of the analysis hinges on eliminating biases, particularly data leakage, through meticulous procedures:

### **Data Preparation**
- **Outlier Detection**: Implemented using IQR thresholds and statistical methods to identify and appropriately handle anomalies
- **Missing Value Imputation**: Transparent treatment of missing data using techniques appropriate to the time series context (forward-fill, interpolation, etc.)
- **Normalization Strategy**: 
  - **Prohibition**: Strictly prohibits global normalization across the entire dataset
  - **Mandate**: Per-series normalization is required to prevent information from the test set influencing the training set
  - **Prevention of Data Leakage**: Ensures that normalization parameters (mean, std) are computed exclusively from the training set

### **Validation Framework**
- **Gold Standard**: Rolling Origin approach (Time-Series Cross-Validation)
  - Iteratively evaluates models on future steps, accurately mimicking real-world deployment
  - Prevents look-ahead bias and respects temporal ordering
  - Provides robust performance estimates across multiple splits
- **Advantage over Fixed Split**: Captures model robustness across different temporal scenarios

### **Baseline Establishment**
- **Naive Forecast Benchmarking**: Every complex model must statistically outperform a simple naive forecast (e.g., seasonal naive $\hat{y}_{t} = y_{t-s}$)
- **Purpose**: Confirms the model is capturing meaningful signal rather than exhibiting random walk-like behavior
- **Statistical Rigor**: Establishes a meaningful point of comparison for all subsequent analyses

---

## 3. Comprehensive Evaluation Framework

Evaluation must be multi-faceted, extending beyond simple point-forecast metrics to include probabilistic assessment and statistical significance testing.

### **Diverse Point Metrics**
A suite of error measures is recommended to assess different error characteristics:
- **RMSE (Root Mean Squared Error)**: Penalizes large errors heavily
- **MAE (Mean Absolute Error)**: Provides a scale-dependent measure of average error
- **MAPE (Mean Absolute Percentage Error)**: Enables scale-relative comparisons

### **Scaled Error Measures**
- **MASE (Mean Absolute Scaled Error)** or **ReIMAE (Relative Mean Absolute Error)**
- **Purpose**: Normalizes performance against the naive benchmark
- **Advantage**: Ensures fair comparison regardless of data scale, making results comparable across different datasets and domains

### **Probabilistic Forecasting Evaluation**
For models like TFT, evaluation must include assessment of both calibration and sharpness:

**Calibration Metrics**:
- **PICP (Prediction Interval Coverage Probability)**: Measures the proportion of actual values falling within the predicted confidence interval
- **Purpose**: Assesses statistical consistency of uncertainty estimates

**Sharpness Metrics**:
- **CRPS (Continuous Ranked Probability Score)**: Evaluates both the accuracy and precision of probabilistic predictions
- **Interpretation**: Lower CRPS indicates better calibrated and sharper forecast distributions

### **Statistical Validation**
Performance claims must be validated using established statistical procedures:

**Pairwise Comparisons**:
- **Diebold-Mariano (DM) Test**: Tests the null hypothesis that two forecasts have equal accuracy
- **Interpretation**: Allows formal statistical testing of performance differences ($p < 0.05$ significance threshold)

**Multiple Model Comparisons**:
- **Friedman Test**: Non-parametric test comparing multiple models
- **Nemenyi Post-hoc Test**: Identifies which specific models differ significantly from one another
- **Combined Approach**: Provides necessary rigor to state that performance differences are statistically significant

---

## 4. Advanced Analytical Techniques

To provide causal insight and intellectual depth, the blueprint requires techniques that delve into the "why" behind the performance.

### **Ablation Studies**
- **Methodology**: Systematically deconstruct complex architectures, especially TFT
- **Process**: Remove key components (e.g., self-attention layers, gating mechanisms) individually
- **Outcome**: Quantify the functional contribution of each module to final performance
- **Purpose**: Identify critical vs. marginal components for model efficiency and interpretability

### **Interpretability Analysis**

**TFT Variable Selection Networks (VSNs)**:
- Assign importance scores to features
- Identify the most influential drivers of demand
- Provide transparency into which variables the model prioritizes

**TFT Temporal Attention Visualization**:
- Reveals which historical time steps the model focuses on
- Uncovers learned dependencies (e.g., seasonality, trend changes)
- Enables domain experts to validate model reasoning

**Cross-Model Feature Importance Comparison**:
- Compare which variables are deemed important across different architectures
- Identify consensus features vs. architecture-specific insights

### **Sensitivity Analysis**
- **Scope**: Examine model robustness across ranges of hyperparameters
- **Example Parameters**:
  - Prophet's `changepoint_prior_scale`: Controls flexibility of trend changes
  - XGBoost's `learning_rate` and `max_depth`: Control regularization and complexity
  - TFT's `dropout_rate` and `attention_heads`: Control regularization and representational capacity
- **Outcome**: Identify stable vs. brittle configurations
- **Purpose**: Characterize model behavior under parameter uncertainty

---

## 5. Expected Contributions and Impact

This comprehensive approach transforms the project from a standard model comparison into a methodologically grounded, evidence-based scientific experiment that will:

✅ **Advance the field**: Provide rigorous empirical evidence on when and why each model excels  
✅ **Enable informed decisions**: Offer practitioners clear guidance on model selection for retail demand forecasting  
✅ **Establish best practices**: Demonstrate state-of-the-art validation, evaluation, and interpretability techniques  
✅ **Support reproducibility**: Document all procedures in sufficient detail for replication and extension  

---

## Project Structure

```
retail_demand_forecasting/
├── data/                            # Storage for raw, interim, and processed data
│   ├── 01_raw/                      # Original, immutable data sources
│   │   └── retail_store_inventory.csv
│   ├── 02_interim/                  # Data after cleaning (outliers, imputation)
│   │   └── preprocessed_features.pkl
│   ├── 03_processed/                # Data used for training and cross-validation (tsCV)
│   │   └── tscv_splits/
│   └── README.md                    # Data preparation notes and schema documentation
├── notebooks/                       # Exploratory analysis, tuning orchestration, and final report generation
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_SARIMA_Model_Tuning.ipynb     
│   ├── 03_Prophet_Model_Tuning.ipynb
│   ├── 04_XGBoost_Model_Tuning.ipynb
│   ├── 05_TFT_Model_Tuning.ipynb
│   └── 06_Comparative_Analysis_Report.ipynb # Final synthesis and visualization of results
├── src/                             # Modular Python source code (core logic)
│   ├── data_pipeline.py             # Handles cleaning, transformation, and per-series normalization
│   ├── features.py                  # Generates lagged features and rolling statistics for XGBoost
│   ├── validation.py                # Implements the rolling origin cross-validation (tsCV) logic
│   ├── evaluation.py                # Contains all error metrics (RMSE, MASE, PICP) and statistical tests
│   └── models/                      # Dedicated classes/wrappers for each forecasting model
├── models/                          # Saved, trained model artifacts
│   ├── sarima_best_params.json
│   ├── prophet_best_model.pkl
│   ├── xgboost_final_model.json
│   ├── tft_checkpoint/
│   └── README.md                    # Documentation on saved model parameters and versions
├── results/                         # Outputs of the rigorous analysis (evidence for the thesis)
│   ├── model_performance_metrics.csv# Final table of all point and probabilistic metrics
│   ├── statistical_tests/           # P-values from Diebold-Mariano and Nemenyi tests
│   ├── interpretability_plots/      # TFT Attention Maps, VSN scores, XGBoost Feature Importance
│   └── sensitivity_analysis/        # Hyperparameter tuning logs
├── .gitignore
├── config.yaml                      # Centralized configuration for hyperparameters and settings
├── requirements.txt                 # All required Python packages and their version numbers
└── METHODOLOGY.md                   # The detailed scholarly blueprint for data preparation, validation, and evaluation
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- See `requirements.txt` for detailed dependencies

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
Execute notebooks in order:
1. `01_EDA_and_Preprocessing.ipynb` - Data exploration and preparation
2. `02_SARIMA_Model_Tuning.ipynb` - SARIMA model development
3. `03_Prophet_Model_Tuning.ipynb` - Prophet model development
4. `04_TFT_Model_Tuning.ipynb` - Temporal Fusion Transformer development
5. `05_Xgboost_Model_Tuning.ipynb` - XGBoost model development
6. `06_Comparative_Analysis_Report.ipynb` - Comprehensive comparative analysis

---

## Key References

- **Box-Jenkins ARIMA**: Box, G. E., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*
- **Prophet**: Taylor, S. J., & Letham, B. (2018). Forecasting at scale
- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- **Temporal Fusion Transformer**: Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Time Series Cross-Validation**: Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series forecasting evaluation

---

## Author
M.A.Chanupa Deshan Munasinghe , Darshan R

## License
MIT License

Copyright (c) 2025 M.A.Chanupa Deshan Munasinghe, Darshan R

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
This project is structured as a research-level comparative study following best practices in forecasting methodology and statistical validation.
