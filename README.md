# Bank Marketing Term Deposit Prediction

## a. Problem Statement
The goal is to predict whether a client will subscribe to a term deposit (`yes` or `no`) based on client demographic information and previous marketing campaign data. This is a binary classification problem.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository (Bank Marketing, ID 222)
- **Instances**: 45,211
- **Features**: 16 (mix of numeric and categorical)
- **Target**: `y` (yes/no – whether the client subscribed to a term deposit)
- **Features include**: age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome
- The dataset is imbalanced (majority class is “no”).

## c. GitHub Repository Link
https://github.com/AKSHAY2026AK05050/bank-marketing-ml-assignment

## d. Models Used

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.8457   | 0.9079 | 0.4182    | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree            | 0.8782   | 0.6933 | 0.4785    | 0.4518 | 0.4648 | 0.3963 |
| KNN                      | 0.8962   | 0.8277 | 0.5990    | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes              | 0.8548   | 0.8101 | 0.4059    | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine   | 0.8456   | 0.9256 | 0.4225    | 0.8715 | 0.5691 | 0.5361 |

### Observations

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Strong overall performance with high AUC and good recall. Balanced metrics make it a reliable choice. |
| Decision Tree            | Decent accuracy but lower AUC and weaker separation of classes compared to linear models. |
| KNN                      | Highest accuracy but lowest recall among the stronger models — tends to miss many positive cases. |
| Naive Bayes              | Moderate performance; assumptions of feature independence limit its effectiveness on this data. |
| Support Vector Machine   | Best AUC and highest MCC. Excellent recall, making it the strongest model for identifying potential subscribers. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** (best AUC and MCC). Logistic Regression is a close second and is faster to train/predict. |

## How to Run Locally
1. Clone the repository
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
