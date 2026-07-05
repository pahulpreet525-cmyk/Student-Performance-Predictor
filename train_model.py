import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("/Users/gurpreet/Downloads/student_performance.csv")

# Remove missing values
data = data.dropna()

# Convert text columns into numbers
label_encoders = {}

for column in data.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    label_encoders[column] = encoder

# Features and Target
X = data[[
    "Hours_Studied",
    "Attendance",
    "Previous_Scores"
]]

y = data["Exam_Score"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/student_model.pkl")

print("✅ Model Trained Successfully!")