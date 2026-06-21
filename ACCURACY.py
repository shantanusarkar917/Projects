import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv(
r"C:\Users\shantanu sarkar\Downloads\final_dataset.csv"
)

# Inputs
X = df[
[
    "Irradiance",
    "Temperature",
    "Expected_Power",
    "Actual_Power",
    "Loss"
]
]

# Output
y = df["Cleaning"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print(f"Accuracy = {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print("\nFeature Importance:")
for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        f"{feature}: {importance:.4f}"
    )

# Save Model
joblib.dump(
    model,
    r"C:\Users\shantanu sarkar\Downloads\solar_cleaning_ai.pkl"
)

print("\nAI Saved Successfully")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(
    "\nCross Validation Accuracy:",
    scores.mean()*100
)