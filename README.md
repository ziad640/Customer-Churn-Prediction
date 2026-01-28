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

## 🛠️ Installation & Usage

### 1️⃣ Clone the Project
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

###2️⃣ Install Dependencies
'''bash
pip install -r requirements.txt

###3️⃣ Run the Entire Pipeline
'''bash
python main.py


###The pipeline will automatically:

Load or generate synthetic data (generate_data.py)

Preprocess data (preprocessing.py)

Run full EDA and generate plots (full_eda.py)

Train the XGBoost model (modeling.py)

Evaluate model performance (evaluate_model.py)

Save the Top 200 at-risk customers (inference.py)

###🎓 About the Project

This project was developed as a Capstone Project for Data Science Revision.
It demonstrates proficiency in end-to-end ML development, from synthetic data generation to business-oriented inference.

Business-focused: Provides actionable insights for telecom retention teams.

Modular & Reusable: Each step is separated into scripts for maintainability and testing.

Production-ready: Can be extended to real customer datasets with minimal changes.

###📊 Reports & Figures

All generated charts from the EDA and model evaluation are stored in:

reports/figures/

###💡 License

MIT License
