import pandas as pd
import joblib

model = joblib.load(
r"C:\Users\shantanu sarkar\Downloads\solar_cleaning_ai.pkl"
)

irradiance = float(input("Irradiance: "))
temperature = float(input("Temperature: "))
expected_power = float(input("Expected Power: "))
actual_power = float(input("Actual Power: "))

loss = (
    (expected_power-actual_power)
    /
    expected_power
)*100

test = pd.DataFrame({
    "Irradiance":[irradiance],
    "Temperature":[temperature],
    "Expected_Power":[expected_power],
    "Actual_Power":[actual_power],
    "Loss":[loss]
})

result = model.predict(test)

print("\nLoss = %.2f%%" % loss)

if result[0] == 1:
    print("⚠ CLEANING REQUIRED")
else:
    print("✅ NO CLEANING REQUIRED")