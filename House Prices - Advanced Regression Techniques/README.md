# House Prices Prediction: Advanced Tabular Data Processing

## Overview
This project demonstrates professional-grade machine learning pipelines for regression tasks on complex tabular data. Rather than just focusing on competitive benchmarks, the primary objective of this repository is to showcase robust data preprocessing, deep feature engineering, and the deployment of advanced gradient boosting algorithms (LightGBM and XGBoost) to extract meaningful patterns from structured datasets.

Achieving a highly accurate model on this dataset serves as a strong foundation in tabular data manipulation before transitioning into more complex domains like Deep Learning, Natural Language Processing (NLP), and Image Processing.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `seaborn`, `matplotlib`
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`

## Workflow & Methodology

### 1. Exploratory Data Analysis (EDA) & Target Transformation
* Analyzed the distribution of the target variable (`SalePrice`).
* Applied a logarithmic transformation (`np.log1p`) to `SalePrice` to normalize the distribution and improve model convergence during regression training.

### 2. Data Cleaning & Imputation
Handled missing values strategically based on feature types:
* **Categorical (None):** Imputed missing values with `"None"` for features where NaN implies the absence of the feature (e.g., `PoolQC`, `Fence`, `GarageType`).
* **Numerical (Zero):** Imputed with `0` for continuous variables representing area or counts (e.g., `GarageArea`, `BsmtFinSF1`).
* **Group-based Imputation:** Filled missing `LotFrontage` values using the median value of the corresponding `Neighborhood`.
* **Mode Imputation:** Filled general categorical features with their statistical mode.

### 3. Feature Engineering
Created highly predictive aggregate features to capture domain knowledge:
* `Total_Square_Footage`: Combined basement, 1st floor, and 2nd floor areas.
* `Total_Bathrooms`: Aggregated full and half bathrooms across all floors.
* `House_Age` & `Years_Since_Remodel`: Engineered temporal features based on the year sold.
* `Total_Porch_Area`: Summed various porch and deck area metrics.

### 4. Categorical Encoding
* **Ordinal Encoding:** Mapped subjective quality assessments (e.g., `Ex`, `Gd`, `TA`, `Fa`, `Po`) to discrete numerical scales to preserve inherent ranking.
* **One-Hot Encoding:** Converted remaining nominal categories (including numeric columns treated as categories like `MSSubClass`) using `pd.get_dummies()`.

### 5. Modeling
* Built regression models utilizing **LightGBM** and **XGBoost** architectures.
* Leveraged their native handling of sparse data and tree-based gradient boosting mechanisms to capture non-linear relationships.

## Results
* **Evaluation Metric Score:** `0.12630` (RMSLE)
* The engineered pipeline efficiently captures the variance in house prices, reflecting a highly tuned tabular data processing methodology.

## Future Exploration
Having established a rigorous framework for tabular data processing, future iterations of this repository may explore:
* Integrating Deep Learning approaches for structured data (e.g., PyTorch-based neural networks).
* Extending these foundational data representation concepts to other unstructured data domains like NLP or computer vision tasks.
