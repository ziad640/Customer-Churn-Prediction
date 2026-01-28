import pandas as pd
from xgboost import XGBClassifier
import joblib

from data_preprocessing import load_raw_data, preprocess

df = preprocess(load_raw_data())
X = df.drop("Churn", axis=1)
y = df["Churn"]

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(len(y[y==0])/len(y[y==1])),  # handle imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Save model
joblib.dump(model, "models/xgb_model.pkl")
