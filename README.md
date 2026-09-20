# Customer Churn Prediction

A machine learning model that predicts customer churn for a telecom company, helping identify at-risk customers before they leave.

## Key Findings (EDA)
 - Contract Type: Month-to-Month customers are significantly more likely to Churn than any other type, because they are not committed enough.
 - Tenure: Churn is heavy in the early tenure of 0-5 months then it drops drastically in later 10-15 months, shows that new custoners are the one to Churn more.
 - Monthly Charges: The Customers who churned had slightly higher monthly charges than the customers who stayed, although the changes are not as high as other finds but something to look into.

 ## Approach
- Cleaned messy TotalCharges column (hidden empty strings)
- Compared 4 classification models — Logistic Regression, Random Forest, KNN, XGBoost
- Addressed class imbalance (73/27 churn split) using class weighting
- Tuned all models with GridSearchCV before final comparison
- Prioritized recall on churn class since missing an at-risk customer costs more than a false alarm(aka false negatives over false positives)

## Final Model

**XGBoost** (tuned via GridSearchCV)
- Recall (Churn): 0.79
- Precision (Churn): 0.51
- F1-score (Churn): 0.62
- Accuracy: 0.74

Selected for the best balance between recall and precision after tuning and fairly comparing all four candidate models (Logistic Regression, Random Forest, KNN, XGBoost).

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook.

## Dataset
Telco Customer Churn — Kaggle (IBM Sample Dataset)

## How to Run
1. Clone the repo
2. pip install -r requirements.txt
3. jupyter notebook
4. Run notebook.ipynb