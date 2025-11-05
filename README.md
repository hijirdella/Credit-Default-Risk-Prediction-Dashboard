Got it — here’s the refined and professional version of your **`README.md`** without emojis or any mention of Bank Danamon.
It keeps a formal tone, focuses on your data science and deployment work, and is GitHub-ready.

---

```markdown
# Credit Default Risk Prediction Dashboard

This repository contains a complete credit default risk modeling pipeline and an interactive Streamlit dashboard designed to demonstrate an end-to-end approach to credit risk analysis — from raw payment data to customer-level prediction.

The project illustrates practical data science workflow components, including data cleaning, feature engineering, model training, evaluation, and deployment in a web-based interface for business users.

---

## Project Overview

**Goal:**  
Predict the likelihood of credit default using historical loan and repayment data at the customer level.

**Components:**
- **Data Preparation & Feature Engineering**  
  Aggregation from payment-level to customer-level features such as delinquency ratios and repayment behavior.
- **Modeling**  
  Multiple machine learning models trained and optimized using cross-validation.
- **Evaluation**  
  AUC, F1, Precision, Recall, and class imbalance handling using SMOTE and undersampling.
- **Deployment (Streamlit)**  
  Interactive web application that allows users to upload data, perform quick EDA, run the trained model, and download predictions.

---

## Features

- Upload raw loan or payment-level data (`combined_df.csv`)
- Automatic feature engineering and aggregation per customer
- Quick exploratory data analysis (EDA)
- Run a trained credit default prediction model (`best_credit_default_model.pkl`)
- View charts for prediction counts, percentages, and feature importance
- Download prediction results as CSV

---

## Repository Structure

```

├── app_credit_default.py        # Streamlit dashboard for EDA and prediction
├── creditdefault.py             # Core functions for model and feature engineering
├── best_credit_default_model.pkl # Trained model file
├── combined_df.csv              # Example raw dataset
└── README.md                    # Project documentation

````

---

## Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/credit-default-dashboard.git
cd credit-default-dashboard
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install manually:

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

### 3. Run the Streamlit dashboard

```bash
streamlit run app_credit_default.py
```

---

## App Workflow

1. Upload your raw dataset (`combined_df.csv` or similar)
2. View automatic EDA summaries (DPD, loan purpose, province, missing data)
3. Generate customer-level predictions using the trained model
4. Visualize default vs non-default distributions
5. Download the prediction results as a CSV file

---

## Example Output

| customer_id   | prob_default | pred_default |
| ------------- | ------------ | ------------ |
| 2003020231588 | 0.999996     | 1            |
| 2003020234027 | 0.999996     | 1            |
| 2003020300323 | 0.000003     | 0            |
| 2003020308160 | 0.000004     | 0            |

**Charts displayed:**

* Bar chart: number of predicted defaults vs non-defaults
* Pie chart: prediction percentages by class
* Feature importance ranking

---

## Model Summary

**Selected Features:**

* `early_payment_ratio`
* `ontime_payment_ratio`
* `late_payment_ratio`
* `total_on_time`
* `repay_installment_ratio`

**Algorithms evaluated:**

* Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and others

**Evaluation metrics:**

* ROC-AUC, Precision, Recall, F1-score, PR-AUC

---

## Business Use

This model can support:

* Credit risk segmentation and early warning systems
* Customer behavior monitoring
* Portfolio performance tracking and decision optimization

The Streamlit interface allows non-technical users to upload data, view insights, and export predictions without writing any code.

---

## Author

**Hijir Della Wirasti**
Data Scientist – Risk Analytics & Credit Scoring

Email: **[hijirdw@gmail.com](mailto:hijirdw@gmail.com)**
LinkedIn: [linkedin.com/in/hijirdella](https://www.linkedin.com/in/hijirdella/)
Portfolio: [hijirdata.com](https://www.hijirdata.com/)

---

## License

This project is distributed for educational and portfolio purposes.
Commercial use requires prior permission from the author.

```


