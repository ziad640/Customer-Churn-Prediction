import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import load_raw_data, preprocess
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = preprocess(load_raw_data())
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model and predict on TEST SET only
model = joblib.load("models/xgb_model.pkl")
y_prob = model.predict_proba(X_test)[:,1]

threshold = 0.4
y_pred = (y_prob >= threshold).astype(int)

print(classification_report(y, y_pred))
print("ROC-AUC:", roc_auc_score(y, y_pred))
