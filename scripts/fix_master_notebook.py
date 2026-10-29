import nbformat
import shutil

backup_path = "AirWatch_Comprehensive_Master.ipynb.bak"
shutil.copyfile("AirWatch_Comprehensive_Master.ipynb", backup_path)
print(f"Backup saved to {backup_path}")

with open("AirWatch_Comprehensive_Master.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Ensure cell 13 has explicit self-contained imports
cell_13_code = """# --- 5. Inspect Engineered Features & Feature Importance ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature-engineered dataset sample
train_data_path = "train_model_dataset.csv"
if os.path.exists(train_data_path):
    df_features = pd.read_csv(train_data_path, nrows=5000)
    print("Engineered Feature Store Columns:")
    print(df_features.columns.tolist()[:25])

# Plot Top Gini Feature Importance from Gradient Boosted Model
fi_path = "data/feature_importance.csv"
if os.path.exists(fi_path):
    df_fi = pd.read_csv(fi_path).head(12)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_fi, x="Importance", y="Feature", palette="rocket")
    plt.title("Top 12 Features for Predicting Next-Day Air Quality Risk (XGBoost / LightGBM)", fontweight="bold")
    plt.xlabel("Gini Split Feature Importance")
    plt.ylabel("Engineered Predictor")
    plt.tight_layout()
    plt.show()
    print(df_fi.to_string(index=False))
"""

nb.cells[13].source = cell_13_code

with open("AirWatch_Comprehensive_Master.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Updated cell 13 safely.")
