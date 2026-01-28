import pandas as pd
import joblib
from data_preprocessing import load_raw_data, preprocess

df = preprocess(load_raw_data())
X = df.drop("Churn", axis=1)

model = joblib.load("models/xgb_model.pkl")
df["churn_prob"] = model.predict_proba(X)[:,1]

# Select top 200 highest probability customers
top_customers = df.sort_values("churn_prob", ascending=False).head(200)
top_customers.to_csv("data/processed/top_200_churners.csv", index=False)
