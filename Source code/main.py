from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the customer churn dataset
df = pd.read_csv(
    "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Display first five records
print("\nCUSTOMER CHURN DATASET")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nDATASET COLUMNS")
print(df.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION")
df.info()


# STEP 2: DATASET UNDERSTANDING

# 1. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 2. Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 3. Check numerical columns
print("\nNUMERICAL COLUMNS")
print(df.select_dtypes(include="number").columns.tolist())

# 4. Check categorical columns
print("\nCATEGORICAL COLUMNS")
print(df.select_dtypes(include=["str"]).columns.tolist())

# 5. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 6. Check customer churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

# 7. Check churn percentage
print("\nCHURN PERCENTAGE")
print(df["Churn"].value_counts(normalize=True) * 100)


# =====================================
# STEP 3: DATA CLEANING AND PREPROCESSING
# =====================================

import numpy as np

# 1. Remove unwanted spaces from TotalCharges
df["TotalCharges"] = df["TotalCharges"].str.strip()

# 2. Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTOTAL CHARGES DATA TYPE")
print(df["TotalCharges"].dtype)

# 3. Check missing values again
print("\nMISSING VALUES AFTER CONVERSION")
print(df.isnull().sum())

# 4. Display rows having missing TotalCharges
print("\nROWS WITH MISSING TOTAL CHARGES")
print(df[df["TotalCharges"].isnull()])

# 5. Handle missing TotalCharges
# Blank TotalCharges records correspond to
# customers with zero tenure in this dataset.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 6. Remove duplicate records
df = df.drop_duplicates()

# 7. Check final missing values
print("\nFINAL MISSING VALUES")
print(df.isnull().sum().sum())

# 8. Check final dataset shape
print("\nCLEANED DATASET SHAPE")
print(df.shape)

# 9. Save cleaned dataset
df.to_csv(
    "Dataset/cleaned_customer_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# =====================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# =====================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("EDA", exist_ok=True)

# Graph 1: Customer Churn Distribution

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

plt.savefig("EDA/churn_distribution.png")
plt.show()

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)

plt.title("Customer Churn Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.savefig("EDA/churn_gender.png")
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn Based on Contract")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.savefig("EDA/churn_contract.png")
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn Based on Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.savefig("EDA/churn_internet.png")
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.savefig("EDA/monthly_charges.png")
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")

plt.savefig("EDA/customer_tenure.png")
plt.show()


# =====================================
# STEP 5: FEATURE AND TARGET SELECTION
# =====================================

# Remove customerID because it is only an ID
X = df.drop(["customerID", "Churn"], axis=1)

# Select Churn as target variable
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Display input features
print("\nINPUT FEATURES (X)")
print(X.head())

# Display target variable
print("\nTARGET VARIABLE (y)")
print(y.head())

# Display dimensions
print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

# Check target distribution
print("\nTARGET DISTRIBUTION")
print(y.value_counts())


# =====================================
# STEP 6: TRAIN TEST SPLIT
# =====================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTESTING DATA")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =====================================
# STEP 7: DATA PREPROCESSING FOR ML
# =====================================

# Identify numerical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["str"]
).columns

# Numerical data preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing setup completed!")


# =====================================
# STEP 8: LOGISTIC REGRESSION
# =====================================

lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train model
lr_model.fit(X_train, y_train)

# Predict test data
lr_predictions = lr_model.predict(X_test)

print("\nLOGISTIC REGRESSION TRAINED SUCCESSFULLY!")

print("First 10 Predictions:")
print(lr_predictions[:10])


# =====================================
# STEP 9: RANDOM FOREST
# =====================================

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train model
rf_model.fit(X_train, y_train)

# Predict test data
rf_predictions = rf_model.predict(X_test)

print("\nRANDOM FOREST TRAINED SUCCESSFULLY!")

print("First 10 Predictions:")
print(rf_predictions[:10])


# =====================================
# STEP 10: MODEL EVALUATION
# =====================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(name, actual, predicted):

    print("\n", name)

    print("Accuracy:",
          accuracy_score(actual, predicted))

    print("Precision:",
          precision_score(actual, predicted))

    print("Recall:",
          recall_score(actual, predicted))

    print("F1 Score:",
          f1_score(actual, predicted))


# Evaluate Logistic Regression
evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_predictions
)

# Evaluate Random Forest
evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Logistic Regression Confusion Matrix
lr_cm = confusion_matrix(y_test, lr_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("EDA/logistic_confusion_matrix.png")
plt.show()

# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("EDA/random_forest_confusion_matrix.png")
plt.show()

print("\nLOGISTIC REGRESSION REPORT")
print(classification_report(y_test, lr_predictions))

print("\nRANDOM FOREST REPORT")
print(classification_report(y_test, rf_predictions))

from sklearn.model_selection import GridSearchCV

# Parameters to test
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

# Create Grid Search
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Train and find the best parameters
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Predictions
best_rf_predictions = best_rf_model.predict(X_test)

# Display results
print("\nBEST RANDOM FOREST PARAMETERS")
print(grid_search.best_params_)

print("\nBEST RANDOM FOREST CROSS-VALIDATION F1 SCORE")
print(grid_search.best_score_)

print("\nIMPROVED RANDOM FOREST RESULTS")
evaluate_model(
    "TUNED RANDOM FOREST",
    y_test,
    best_rf_predictions
)

import pandas as pd
import matplotlib.pyplot as plt

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, lr_predictions),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test, best_rf_predictions)
    ],
    "Precision": [
        precision_score(y_test, lr_predictions),
        precision_score(y_test, rf_predictions),
        precision_score(y_test, best_rf_predictions)
    ],
    "Recall": [
        recall_score(y_test, lr_predictions),
        recall_score(y_test, rf_predictions),
        recall_score(y_test, best_rf_predictions)
    ],
    "F1 Score": [
        f1_score(y_test, lr_predictions),
        f1_score(y_test, rf_predictions),
        f1_score(y_test, best_rf_predictions)
    ]
})

print("\nFINAL MODEL COMPARISON")
print(comparison)

comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Customer Churn Prediction - Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("EDA/model_comparison.png")
plt.show()

import os
import joblib

# Create folder if it does not exist
os.makedirs("Trained Model", exist_ok=True)

# Save the best model
joblib.dump(
    best_rf_model,
    "Trained Model/customer_churn_model.pkl"
)

print("\nFINAL MODEL SAVED SUCCESSFULLY!")