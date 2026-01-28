import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(n_estimators=300, max_depth=5, scale_pos_weight=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test, threshold=0.35):
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nClassification Report (Threshold={threshold}):")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    X = df.drop(columns='Churn')
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test)
    save_model(rf, "models/random_forest.pkl")

    # Train XGBoost
    xgb = train_xgboost(X_train, y_train)
    evaluate_model(xgb, X_test, y_test)
    save_model(xgb, "models/xgboost.pkl")
