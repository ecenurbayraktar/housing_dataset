# House Price Prediction with Advanced Machine Learning

## Project Overview

This project predicts house sale prices using the Ames Housing dataset. The workflow covers the complete machine learning pipeline, from data preprocessing and feature engineering to model optimization, explainability, residual analysis, and final test predictions.

The objective is to build a robust regression model while following good machine learning practices such as proper preprocessing, cross-validation, hyperparameter optimization, and model interpretation.

---

## Dataset

* **Training data:** `train.csv`
* **Test data:** `test.csv`
* **Target variable:** `SalePrice`

---

## Project Workflow

### 1. Data Exploration

* Dataset inspection
* Missing value analysis
* Duplicate value check
* Data type analysis
* Numerical and categorical feature identification
* Correlation analysis
* Outlier visualization

---

### 2. Missing Value Handling

Different imputation strategies were applied depending on the feature type.

* KNN Imputation for numerical features
* Domain-based filling for structural missing values (e.g. `None`)
* Median imputation inside optimization pipelines
* Most frequent imputation for categorical features

---

### 3. Feature Engineering

Several new features were created to improve model performance.

Examples include:

* Garage availability
* Basement availability
* Pool availability
* Fireplace availability
* Property age related features
* Garage related features
* House renovation indicators
* Seasonal information
* Additional binary indicator features

---

### 4. Feature Encoding

Multiple encoding techniques were applied depending on feature characteristics.

* Ordinal Encoding
* One-Hot Encoding
* Label Encoding
* Rare Category Encoding
* Cyclic / Seasonal Feature Engineering

---

### 5. Feature Scaling

Numerical features were standardized before model training where appropriate.

---

## Machine Learning Models

The following regression models were trained and compared:

* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor
* LightGBM Regressor

Model performance was evaluated using:

* Cross Validation
* RMSE
* R² Score

---

## Hyperparameter Optimization

A two-stage optimization strategy was used.

### Randomized Search

RandomizedSearchCV was first used to efficiently explore the hyperparameter space.

### Grid Search

The best LightGBM parameters obtained from Randomized Search were further refined using GridSearchCV.

Finally, the performance of:

* Randomized Search
* Randomized Search + Grid Search

was compared to measure the improvement obtained after fine-tuning.

---

## Model Interpretation

To better understand model behavior, two explainability techniques were used.

### Feature Importance

Feature importance was generated for:

* CatBoost
* XGBoost
* LightGBM

allowing comparison of the most influential variables.

### SHAP Analysis

SHAP (SHapley Additive exPlanations) was applied to the final LightGBM model to explain:

* Global feature importance
* Individual feature contributions
* Model decision behavior

Visualizations include:

* SHAP Beeswarm Plot
* SHAP Feature Importance Plot

---

## Residual Analysis

Residual analysis was performed using cross-validation predictions.

The following visualizations were generated:

* Residual vs Predicted Plot
* Residual Distribution Histogram

These plots were used to evaluate:

* Prediction bias
* Error distribution
* Model stability
* Potential heteroscedasticity

---

## Final Model

The final model is the optimized **LightGBM Regressor**, selected after comparing all candidate models.

Final predictions are generated on the unseen test dataset and exported as a CSV submission file.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* SHAP

---

## Project Highlights

* Complete end-to-end machine learning pipeline
* Advanced feature engineering
* Multiple encoding strategies
* Cross-validation based evaluation
* Hyperparameter optimization with Randomized Search and Grid Search
* Model comparison
* Feature importance analysis
* SHAP explainability
* Residual diagnostics
* Final prediction generation for unseen test data
