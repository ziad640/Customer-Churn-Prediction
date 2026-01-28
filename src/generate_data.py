import pandas as pd
import numpy as np
import os

def generate_telecom_data(n_samples: int = 100000, seed: int = 42) -> None:
    """
    Generates a synthetic telecom churn dataset with realistic business logic.
    """
    np.random.seed(seed)
    
    # 1. Generate Base Features
    monthly_bill: np.ndarray = np.random.uniform(30, 150, n_samples)
    total_data_usage: np.ndarray = np.random.uniform(5, 500, n_samples)
    service_calls: np.ndarray = np.random.poisson(1.5, n_samples)
    
    # 0 = Monthly, 1 = Yearly (60/40 split)
    contract_type: np.ndarray = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # 2. Logic-Based Churn Probability (Log-odds / Logit)
    logit: np.ndarray = (
        0.05 * (monthly_bill / 50) +      
        0.8 * (service_calls > 3) +       
        -0.5 * (total_data_usage / 100) + 
        1.2 * (contract_type == 0) +      
        -3.0                              
    )
    
    # Convert logit to probability
    prob: np.ndarray = 1 / (1 + np.exp(-logit))
    
    # Generate binary target (1=Churn, 0=Stay)
    churn: np.ndarray = np.random.binomial(1, prob)
    
    df: pd.DataFrame = pd.DataFrame({
        'Monthly_Bill': monthly_bill.round(2),
        'Total_Data_Usage': total_data_usage.round(2),
        'Customer_Service_Calls': service_calls,
        'Contract_Type': contract_type,
        'Churn': churn
    })
    
    # Ensure the directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/raw/telecom_churn.csv', index=False)
    print(f"✅ Dataset generated: 100,000 rows. Churn rate: {df.Churn.mean():.2%}")

if __name__ == "__main__":
    generate_telecom_data()