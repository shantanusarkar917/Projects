import joblib
import pandas as pd

# Load AI Model
model = joblib.load(
r"C:\Users\shantanu sarkar\Downloads\solar_cleaning_ai.pkl"
)

# Feature names
features = [
    "Irradiance",
    "Temperature",
    "Expected_Power",
    "Actual_Power",
    "Loss"
]

print("\n===== FEATURE IMPORTANCE =====\n")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(
        f"{feature}: {importance*100:.2f}%"
    )