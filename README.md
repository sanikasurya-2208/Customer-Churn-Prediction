# Customer Churn Prediction Using Machine Learning

## 1. Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a company's services.

The system analyzes customer information such as tenure, monthly charges, contract type, internet service and payment method to predict customer churn.

The project uses machine learning classification algorithms and provides an interactive web application using Streamlit.

## 2. Objectives

- To understand customer churn using machine learning.
- To analyze customer data and identify important patterns.
- To preprocess and clean the dataset.
- To train and compare different classification algorithms.
- To evaluate model performance using different metrics.
- To develop an interactive web application for churn prediction.

## 3. Dataset

**Dataset Name:** IBM Telco Customer Churn Dataset

**Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- Total records: 7,043
- Total columns: 21
- Target variable: Churn
- Input features: 19 (after removing customerID and separating Churn)

The dataset contains customer information, services, account details and churn status.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the customer churn dataset.
2. Checked missing values and duplicate records.
3. Converted TotalCharges into numeric format.
4. Handled missing TotalCharges values.
5. Removed duplicate records.
6. Separated input features (X) and target variable (y).
7. Applied One-Hot Encoding to categorical features.
8. Applied Standard Scaling to numerical features.
9. Split the dataset into training and testing data.

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed using visualizations to understand:

- Customer churn distribution.
- Churn based on gender.
- Churn based on contract type.
- Churn based on internet service.
- Monthly charges distribution.
- Customer tenure distribution.

## 7. Machine Learning Algorithms

Three model configurations were evaluated:

### A. Logistic Regression
A classification algorithm used to predict the probability of customer churn.

### B. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to make predictions.

### C. Tuned Random Forest
Random Forest was optimized using GridSearchCV with different hyperparameter combinations.

Best parameters:
- n_estimators: 200
- max_depth: 10
- min_samples_split: 5

## 8. Model Evaluation

The models were evaluated using Accuracy, Precision, Recall and F1 Score.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Random Forest | 77.00% | 55.92% | 63.10% | 59.30% |
| Tuned Random Forest | 75.09% | 52.14% | 74.87% | 61.47% |

The Tuned Random Forest achieved the highest F1 Score among the three evaluated models and was selected as the final model.

## 9. Web Application

A web application was developed using Streamlit.

The application allows users to:

- Enter customer information.
- Submit customer details for prediction.
- View the predicted churn status.
- View estimated churn probability.
- View model performance metrics.

## 10. Project Structure

```text
Customer-Churn-Prediction/
│
├── Dataset/
│   ├── original dataset CSV
│   └── cleaned_customer_churn.csv
│
├── EDA/
│   └── visualization images
│
├── Source code/
│
├── Trained Model/
│   └── customer_churn_model.pkl
│
├── main.py
├── app.py
├── app_backup.py
├── requirements.txt
└── README.md
```

## 11. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project folder

```bash
cd Customer-Churn-Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

### Step 5: Install required libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

**Note:** The trained model file must be available inside the Trained Model folder before running the application.

To retrain the models and regenerate the saved model, run:

```bash
python main.py
```

## 12. Conclusion

The Customer Churn Prediction project demonstrates how machine learning can be used to identify customers who may leave a service.

Different classification algorithms were trained and compared. The Tuned Random Forest model achieved the highest F1 Score of 61.47% among the evaluated models.

The Streamlit application provides a simple interface for entering customer details and obtaining churn predictions.

## 13. Future Scope

- Improve prediction performance through further experimentation.
- Add more machine learning algorithms.
- Improve the user interface.
- Add customer retention recommendations.
- Deploy the application online.
