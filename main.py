import pandas as pd

# Load the dataset
data = pd.read_csv("/Users/gurpreet/Downloads/student_performance.csv")

print("=" * 50)
print("STUDENT PERFORMANCE DATASET ANALYSIS")
print("=" * 50)

# Dataset shape
print("\n1. Dataset Shape")
print(data.shape)

# Column names
print("\n2. Column Names")
print(data.columns.tolist())

# Data types
print("\n3. Data Types")
print(data.dtypes)

# Missing values
print("\n4. Missing Values")
print(data.isnull().sum())

# Statistical summary
print("\n5. Statistical Summary")
print(data.describe())
