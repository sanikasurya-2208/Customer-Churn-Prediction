
import pandas as pd

# Load original dataset
df = pd.read_csv(
    "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("ORIGINAL DATASET")
print("Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Replace missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove duplicate records
df = df.drop_duplicates()

# Check cleaned dataset
print("\nCLEANED DATASET")
print("Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(
    "Dataset/cleaned_customer_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

