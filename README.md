# Customer Churn Prediction: Logistic Regression

## 📌 Project Overview
This project implements a Machine Learning pipeline using **Logistic Regression** to predict customer churn in a telecommunications dataset. By analyzing customer usage patterns, plan details, and service calls, the model identifies which customers are at high risk of canceling their service. 

Beyond simply predicting churn, this project focuses on **interpretability**—extracting business insights (Odds Ratios) to understand *why* customers leave, allowing businesses to take proactive retention measures.

## 📊 Dataset
* **File:** `churn-bigml-80.csv`
* **Target Variable:** `Churn` (Boolean/Binary)
* **Features:** Includes usage minutes (day/evening/night/international), financial charges, active plans (international/voicemail), and customer service interactions.

## 🛠️ Technologies Used
* **Python 3.x**
* **Pandas & NumPy:** Data manipulation and mathematical operations.
* **Scikit-Learn:** Machine learning (data splitting, scaling, model training, and evaluation).
* **Matplotlib:** Data visualization (ROC Curve).

## 🚀 Key Steps in the Pipeline
1. **Data Preprocessing:** * Converts categorical variables (`Yes`/`No`, `True`/`False`) into machine-readable binary integers (`1`/`0`).
   * Drops noisy or high-cardinality features (like US `State` and `Area code`) to establish a clean baseline model.
2. **Stratified Splitting & Scaling:** * Splits data into 80% training and 20% testing sets, using stratification to maintain the natural imbalance of churners.
   * Standardizes all features using `StandardScaler` so coefficients can be compared fairly.
3. **Model Training:** Fits a `LogisticRegression` algorithm to the scaled training data.
4. **Business Insights (Odds Ratios):** Exponentiates the mathematical coefficients to calculate human-readable Odds Ratios, revealing the top drivers of customer churn.
5. **Evaluation:** Grades the model using Accuracy, Precision, Recall, and plots an interactive Receiver Operating Characteristic (ROC) curve to visualize performance.

## 📈 Key Findings (Example Insights)
Based on the Odds Ratios extracted from the model, we found that:
* **Customer Service Calls** are the highest driver of churn. The likelihood of a customer leaving roughly doubles with an increase in service calls.
* **International Plans** positively correlate with churn, suggesting potential dissatisfaction with international pricing or quality.
* **Voice Mail Plans** act as a strong retention factor, significantly reducing the likelihood of churn.

## 💻 How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
