# Spaceship Titanic: Advanced Ensemble Modeling & Tabular Data Mastery

## Overview
This repository represents a capstone project in advanced tabular data processing and classification. It moves beyond standard exploratory analysis by implementing highly complex, conditional imputation strategies and a robust Stacking Ensemble architecture. 

Mastering these sophisticated data manipulation and ensemble techniques solidifies a deep, professional-grade understanding of structured data. This serves as the final foundational milestone before transitioning into high-dimensional, unstructured data domains such as Deep Learning, Computer Vision, and Image Processing.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Ensembling:** `scikit-learn` (`StackingClassifier`, `RandomForestClassifier`, `LogisticRegression`)
* **Advanced Gradient Boosting:** `xgboost` (`XGBClassifier`), `catboost` (`CatBoostClassifier`)

## Workflow & Methodology

### 1. Intelligent Conditional Imputation
Rather than relying on basic statistical fills (mean/median), missing values were imputed using domain logic and cross-feature relationships:
* **CryoSleep vs. Spending:** Passengers in `CryoSleep` were logically restricted to `0` spending across all amenities (RoomService, Spa, etc.). Conversely, passengers with missing `CryoSleep` statuses but positive amenity spending were imputed as awake (`False`).
* **Group-Based Inference:** Missing `HomePlanet` values were intelligently inferred by looking at the `HomePlanet` of other passengers sharing the same travel `Group` (extracted from `PassengerId`).

### 2. Deep Feature Engineering
* **Cabin Parsing:** Split the highly cardinal `Cabin` string into three distinct, actionable features: `Deck`, `Num`, and `Side`.
* **Group Dynamics:** Engineered a `Group_Size` feature based on the frequency of shared `PassengerId` prefixes, capturing the survival dynamics of traveling parties.
* **Financial Aggregation:** Created a `Total_Spend` feature aggregating all individual amenity expenditures.

### 3. Stacking Ensemble Architecture
To maximize predictive accuracy and generalization, a multi-layer Stacking Classifier was developed:
* **Base Learners:** 
  * `RandomForestClassifier` (handling complex interactions via bagging)
  * `XGBClassifier` (hyperparameter-tuned tree boosting)
  * `CatBoostClassifier` (optimized for categorical depth and performance)
* **Meta-Learner:** A `LogisticRegression` model was trained on the predictions of the base learners to output the final Transported probability.

## Results
* **Evaluation Metric Score:** `0.80687` (Accuracy)
* The combination of logic-driven data cleaning and a multi-model stacking architecture proved highly effective in capturing the non-linear nuances of the dataset.

## Future Outlook
With advanced structured data pipelines, conditional imputation, and ensemble modeling now fully established, future development will pivot entirely toward unstructured data. Upcoming projects will focus on **Deep Learning** architectures, leveraging frameworks like **PyTorch** for **Image Processing** and neural network design.
