import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# 1. READ CSV
# -------------------------------
file_path = os.path.join("Data", "Cleansed", "MLModelvsHPCIResults.csv")

df = pd.read_csv(file_path, sep=",")


print("Columns:", df.columns)
print(df.head())

# -------------------------------
# 2. SORT
# -------------------------------
df = df.sort_values(by="RMSE")
print(df.head())

# -------------------------------
# 3. PLOT RMSE + MAE
# -------------------------------
models = df["Model"]
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(x - width/2, df["RMSE"], width, label="RMSE (£)")
plt.bar(x + width/2, df["MAE"], width, label="MAE (£)")

plt.xlabel("Models")
plt.ylabel("Error (£)")
plt.title("Model Comparison: RMSE vs MAE (Real Price)")
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# -------------------------------
# 4. PLOT R2
# -------------------------------
plt.figure(figsize=(12, 6))

plt.bar(df["Model"], df["R2"])

plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Comparison: R² Score (Real Price)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')

plt.tight_layout()
plt.show()