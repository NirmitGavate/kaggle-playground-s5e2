# Kaggle Playground Series - Backpack Price Prediction (2025)
Top 29% Solution
## Overview
This project is part of the 2025 Kaggle Playground Series, where the objective is to predict the price of backpacks given various attributes. The dataset is synthetic but closely mimics real-world data. The competition provides an opportunity to experiment with different machine learning models and feature engineering techniques.

## Goal
The main goal is to build a predictive model that accurately estimates the price of backpacks based on given features. The competition evaluates submissions using Root Mean Squared Error (RMSE).

## Evaluation Metric
Submissions are scored based on RMSE (Root Mean Squared Error):

## Model Performance

### Training Data Results:

| Experiment | Validation Strategy | Model | Encoding | Feature Engineering | RMSE Score |
|------------|---------------------|-------|----------|----------------------|------------|
| Report-1  | Train-Test Split (20%) | Linear Regression | Label Encoding + One-Hot Encoding | None | 39.16099 |
| Report-2  | Train-Test Split (20%) | XGBRegressor | One-Hot Encoding | None | 39.14936 |
| Report-3  | KFold (5 splits) | XGBRegressor | One-Hot Encoding | None | 39.14512 |
| Report-4  | KFold (5 splits) | XGBRegressor | One-Hot Encoding | Basic Feature Engineering | 39.12372 |
| Report-5  | KFold (10 splits) | LGBMRegressor | One-Hot Encoding | Basic Feature Engineering | 39.11194 |
| Report-6  | KFold (10 splits) | LGBMRegressor | Target Encoding | Basic Feature Engineering | 39.10660 |
| Report-7  | Train-Test Split | Basic Neural Network | Target Encoding | None | 39.10660 |

## Approach

### Data Preprocessing:
- Filled numerical null values using mean/median.
- Replaced missing values in categorical features with the most frequent value or "missing" label.

### Feature Engineering:
- Applied basic feature engineering on numeric columns.
- Used One-Hot Encoding and Target Encoding for categorical columns.

### Model Training:
- Experimented with different models: Linear Regression, XGBoost, LightGBM, and a simple Neural Network.
- Compared performance using different validation strategies:
  - Train-test split (20%)
  - K-Fold cross-validation (5 and 10 splits)

### Hyperparameter Tuning:
- Default model settings were used in initial experiments.
- Further tuning (e.g., adjusting tree depth, learning rate) could improve performance.

### Validation Strategy:
- Used cross-validation (KFold) for better generalization.
- Ensured robustness by evaluating different data splitting techniques.

## Conclusion
- The best RMSE score achieved was **39.10660** using LGBMRegressor with target encoding and basic feature engineering.
- Feature engineering and proper validation strategies significantly improved model performance.
- Future improvements could involve advanced hyperparameter tuning, feature selection, and deep learning models.

## Acknowledgments
- Kaggle Playground Series Team for providing the dataset.
- The Kaggle community for insights and discussions.
- XGBoost and LightGBM for efficient modeling and training.

