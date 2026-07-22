# Titanic Passenger Survival: Professional Tabular Classification

## Overview
This repository focuses on building a rigorous end-to-end classification pipeline for structured tabular data. Developing long-term professional expertise in machine learning requires a solid grasp of feature engineering and classification metrics. Consolidating these skills on complex tabular datasets serves as the final foundational step before transitioning into unstructured, high-dimensional data domains such as Image Processing and Natural Language Processing (NLP).

## Tech Stack
* **Language:** Python
* **Data Manipulation & Processing:** `pandas`, `numpy`
* **Machine Learning & Tuning:** `scikit-learn` (`GridSearchCV`, `train_test_split`, `metrics`)
* **Algorithm:** `xgboost` (`XGBClassifier`)

## Workflow & Methodology

### 1. Advanced Feature Engineering
Extracted hidden patterns within the raw data to improve model interpretability and predictive power:
* **Title Extraction:** Utilized Regular Expressions (Regex) to extract passenger titles (e.g., Mr., Mrs., Miss.) from the raw `Name` string, capturing underlying social status and gender demographics.
* **Family Size Aggregation:** Combined the `SibSp` (siblings/spouses) and `Parch` (parents/children) features into a unified `FamilySize` metric to evaluate the survival odds of solo travelers versus large families.

### 2. Data Cleaning & Imputation
Implemented strategic imputation techniques rather than dropping valuable rows:
* **Age Imputation:** Filled missing `Age` values using the statistical mean of the dataset.
* **Intelligent Categorical Imputation:** Handled missing `Embarked` (port of embarkation) values by grouping the data by `Pclass` (passenger class) and filling the nulls with the mode of that specific class.
* **Dimensionality Reduction:** Dropped highly sparse or noisy columns (`Cabin`, `Ticket`, `Name`, `PassengerId`) to reduce overfitting risks.

### 3. Categorical Encoding
* Applied **One-Hot Encoding** (`pd.get_dummies`) to transform categorical text data into a machine-readable numeric format.
* Used the `drop_first=True` parameter to avoid the dummy variable trap (multicollinearity).

### 4. Modeling & Hyperparameter Tuning
* Deployed an **XGBoost Classifier**, optimized for binary classification (`logloss` evaluation metric).
* Executed an exhaustive **Grid Search with 5-fold Cross-Validation** (`GridSearchCV`) to find the optimal hyperparameters across:
  * Tree complexity (`max_depth`)
  * Number of boosting rounds (`n_estimators`)
  * Step size shrinkage (`learning_rate`)
  * Row and column sampling ratios (`subsample`, `colsample_bytree`)

## Results
The hyperparameter-tuned XGBoost model delivered strong classification performance on the unseen test set:
* **Accuracy:** `~83.86%`
* **Confusion Matrix Breakdown:**
  * True Negatives (Correctly predicted non-survival): `119`
  * True Positives (Correctly predicted survival): `68`
  * False Positives: `15`
  * False Negatives: `21`

## Next Steps
With tabular data processing, imputation, and classification pipelines now thoroughly established, the logical progression is to shift focus toward unstructured data. Future projects will pivot toward **Image Processing** and deep learning architectures.
