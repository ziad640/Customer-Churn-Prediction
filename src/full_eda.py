# full_eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telecom_churn.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Ensure folders exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
df = pd.read_csv(PROCESSED_DATA_PATH)

# ====== EDA ======
# 1. Basic stats
print("Data shape:", df.shape)
print(df.describe())
print("\nChurn distribution:\n", df['Churn'].value_counts(normalize=True))

# 2. Churn count plot
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.savefig(os.path.join(FIGURES_DIR, "churn_count.png"))
plt.close()

# 3. Histograms for numeric features
numeric_cols = ['Monthly_Bill', 'Total_Data_Usage', 'Customer_Service_Calls']
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"{col} Distribution")
    plt.savefig(os.path.join(FIGURES_DIR, f"{col}_hist.png"))
    plt.close()

# 4. Boxplots vs Churn
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.savefig(os.path.join(FIGURES_DIR, f"{col}_boxplot.png"))
    plt.close()

# 5. Contract_Type vs Churn
plt.figure(figsize=(6,4))
sns.countplot(x='Contract_Type', hue='Churn', data=df)
plt.title("Contract_Type vs Churn")
plt.savefig(os.path.join(FIGURES_DIR, "contract_type_churn.png"))
plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation")
plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
plt.close()

print(f"All figures saved in {FIGURES_DIR}")
