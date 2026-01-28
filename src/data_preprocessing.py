import os
import pandas as pd
from pandas import DataFrame

def load_raw_data(path: str = "data/raw/telecom_churn.csv") -> DataFrame:
    """
    Load raw CSV data
    """
    df = pd.read_csv(path)
    return df

def preprocess(df: DataFrame) -> DataFrame:
    """
    Preprocess the raw dataframe
    """
    df['Contract_Type'] = df['Contract_Type'].astype(int)
    return df

def save_processed(df: DataFrame, path: str = "data/processed/processed_data.csv") -> None:
    """
    Save processed dataframe to CSV
    """
    # تأكد ان الفولدر موجود
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")

# --- تشغيل مثال ---
if __name__ == "__main__":
    df = load_raw_data("data/raw/telecom_churn.csv")
    df_processed = preprocess(df)
    save_processed(df_processed, "data/processed/processed_data.csv")
