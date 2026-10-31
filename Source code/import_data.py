
import pandas as pd
import mysql.connector

# Load dataset
df = pd.read_csv(
    "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
).fillna(0)

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=input("Enter MySQL root password: "),
    database="customer_churn_db"
)

cursor = connection.cursor()

# Insert customer records
query = """
INSERT INTO Customers VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s
)
"""

records = [
    tuple(value.item() if hasattr(value, "item") else value
          for value in row)
    for row in df.itertuples(index=False, name=None)
]

cursor.executemany(query, records)
connection.commit()

print("Dataset imported successfully!")
print("Total customer records:", cursor.rowcount)

cursor.close()
connection.close()
