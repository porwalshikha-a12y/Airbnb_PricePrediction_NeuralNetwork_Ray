import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
df = pd.read_csv("Data/Cleansed/airbnb_model_input.csv")

def get_project_output_path(filename=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output/NormalEDA")
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        return os.path.join(output_dir, filename)

    return output_dir

print(df.shape)
print(df.head())


print("=== DATA LOADED ===")
print(df.shape)


# 2. BASIC INFO

print("\n=== HEAD ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())


# 3. NULL ANALYSIS

print("\n=== NULL VALUES ===")
nulls = df.isnull().sum()
print(nulls[nulls > 0])

# % nulls
null_percent = (df.isnull().sum() / len(df)) * 100
print("\n=== NULL % ===")
print(null_percent[null_percent > 0])

# Heatmap of nulls
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap",fontsize=16)

path = get_project_output_path("Missing Values Heatmap.png")
plt.savefig(path, bbox_inches="tight")
print(f"Saved: {path}")
plt.close()


# 4. DATA TYPES

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

print("\nNumerical columns:", list(num_cols))
print("\nCategorical columns:", list(cat_cols))


# 5. DISTRIBUTIONS

for col in num_cols[:10]:  # limit for readability
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}", fontsize=16)

    path = get_project_output_path(f"Distribution of {col}.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()



# 6. CORRELATION

plt.figure(figsize=(12,8))
corr = df[num_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 6}
)
plt.title("Correlation Heatmap", fontsize=16)
path = get_project_output_path(f"Correlation Heatmap.png")
plt.savefig(path, bbox_inches="tight")
print(f"Saved: {path}")
plt.close()


# 7. TARGET ANALYSIS (if price exists)

if 'price' in df.columns:
    plt.figure()
    sns.histplot(df['price'], kde=True)
    plt.title("Price Distribution", fontsize=16)

    path = get_project_output_path(f"Price Distribution.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    plt.figure()
    sns.boxplot(x=df['price'])
    plt.title("Price Boxplot (Outliers)", fontsize=16)

    path = get_project_output_path(f"Price Boxplot (Outliers).png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# 8. CATEGORICAL ANALYSIS

for col in cat_cols[:5]:  # limit
    plt.figure(figsize=(8,4))
    df[col].value_counts().head(10).plot(kind='bar')
    plt.title(f"Top Categories in {col}", fontsize=16)
    plt.xticks(rotation=45)

    path = get_project_output_path(f"Top Categories in {col}.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()




# 9. PAIRPLOT (small subset)

sample_cols = list(num_cols[:5])
pair_df = df[sample_cols].dropna()

sns.pairplot(pair_df)

path = get_project_output_path("pairplot_numeric_subset.png")
plt.savefig(path, bbox_inches="tight")
print(f"Saved: {path}")
plt.close()

# 10. SAVE CLEAN SUMMARY

df.describe().to_csv("eda_summary.csv")

print("\n=== EDA COMPLETE ===")

import matplotlib.pyplot as plt

# Real price metrics
labels = ["RMSE (£)", "MAE (£)"]
values = [50.29, 34.44]

# Create plot
plt.figure()
plt.bar(labels, values)

# Add value labels on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}", ha='center')

# Titles and labels
plt.title("Model Performance (Real Price Scale)")
plt.ylabel("Error (£)")

# Display
plt.tight_layout()
plt.show()