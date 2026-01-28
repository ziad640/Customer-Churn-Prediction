# 📡 Telecom Customer Churn Prediction

## 🏢 Business Case
In the telecom industry, retaining an existing customer is **5x cheaper** than acquiring a new one. This project provides a machine learning solution to identify high-risk customers **before they leave**.

**Objective:**  
Predict which of the 100,000 customers will churn next month and provide a **prioritized list** for the retention team to contact.

---

## 🚀 Key Features & Highlights

- **Production-Ready Pipeline:** Modular system triggered by a single `main.py` orchestrator.  
- **Imbalance Handling:** Utilized `scale_pos_weight` and stratified sampling to address the **15/85 churn distribution**.  
- **Business Intelligence:** Outputs a **Top 200 Lead List** sorted by churn probability for targeted marketing.  
- **Interpretability:** Focuses on **Recall over Accuracy** to minimize the "Lost Customer" cost.

---

## 📂 Project Structure

├── main.py # Automated Pipeline Orchestrator
├── requirements.txt # Environment dependencies
├── src/ # Source Logic
│ ├── generate_data.py # Synthetic data engine (100k samples)
│ ├── preprocessing.py # Feature engineering & cleaning
│ ├── full_eda.py # Automated visualization suite
│ ├── modeling.py # XGBoost & Random Forest training
│ └── inference.py # Priority lead generation
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Model-ready data & Top 200 CSV
├── models/ # Serialized model binaries (.pkl)
└── reports/
└── figures/ # Auto-generated EDA & Performance charts

---

## 🧪 Results & Insights

### 1️⃣ Feature Importance
Based on analysis, the **top drivers for churn** are:

| Feature                  | Insight |
|---------------------------|---------|
| **Contract Type**         | Customers on monthly contracts are significantly more likely to leave. |
| **Total Data Usage**      | Declining usage is a precursor to churn. |
| **Customer Service Calls**| >3 calls significantly increases churn probability (~60% jump). |

### 2️⃣ Model Performance

| Metric                         | Score |
|--------------------------------|-------|
| Recall (Churners)               | 88%   |
| ROC-AUC                         | 0.84  |
| Precision (at 0.4 threshold)   | 74%   |

> **Note:** Threshold 0.4 chosen based on EDA to balance precision and recall for business decisions.

---
