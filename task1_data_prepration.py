
# ===============================
# Task 1: Data Preparation
# Internship: Saiket Systems
# ===============================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 2: Create a sample dataset (since real dataset is not provided)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, np.nan, 22, 28],            # np.nan = missing value
    'Gender': ['F', 'M', 'M', 'M', np.nan],     # categorical + missing
    'Salary': [50000, 60000, 55000, np.nan, 62000],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Step 3: Check dataset information and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values in each column:")
print(df.isnull().sum())

# Step 4: Handle missing values
# Fill numerical columns (Age, Salary) with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Fill categorical columns (Gender) with mode (most frequent value)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

print("\nDataset after handling missing values:")
print(df)

# Step 5: Encode categorical columns
le = LabelEncoder()
categorical_cols = ['Name', 'Gender', 'Department']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nDataset after encoding categorical variables:")
print(df)

# Step 6: Save cleaned dataset to CSV
df.to_csv('cleaned_sample_data.csv', index=False)
print("\nData preparation completed. Cleaned file saved as 'cleaned_sample_data.csv'.") 
from sklearn.model_selection import train_test_split
X = df.drop("Department", axis=1)
y = df["Department"] 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# ===============================
# Task 1: Data Preparation
# Internship: Saiket Systems
# ===============================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 2: Create a sample dataset (since real dataset is not provided)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, np.nan, 22, 28],            # np.nan = missing value
    'Gender': ['F', 'M', 'M', 'M', np.nan],     # categorical + missing
    'Salary': [50000, 60000, 55000, np.nan, 62000],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Step 3: Check dataset information and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values in each column:")
print(df.isnull().sum())

# Step 4: Handle missing values
# Fill numerical columns (Age, Salary) with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Fill categorical columns (Gender) with mode (most frequent value)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

print("\nDataset after handling missing values:")
print(df)

# Step 5: Encode categorical columns
le = LabelEncoder()
categorical_cols = ['Name', 'Gender', 'Department']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nDataset after encoding categorical variables:")
print(df)

# Step 6: Save cleaned dataset to CSV
df.to_csv('cleaned_sample_data.csv', index=False)
print("\nData preparation completed. Cleaned file saved as 'cleaned_sample_data.csv'.")

