import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("cleaned_sample_data.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df[["Age", "Salary"]] = imputer.fit_transform(df[["Age", "Salary"]])

# Define features and target
X = df.drop("Salary", axis=1)   # Features
y = df["Salary"]                # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)
