# ChurnSense — Telecom Customer Churn (Deep Learning)

A self-contained **proof of concept** for telecom churn prediction and retention analytics. The project trains a PyTorch multilayer perceptron (MLP) to predict a customer's likelihood of leaving, using a reproducible data preparation and evaluation pipeline. Includes a Streamlit dashboard to explore high-risk customers.

## What is implemented

- Source ingestion, schema checks, missing value handling, numeric scaling and one-hot encoding.
- Stratified train / validation / holdout-test split (70 / 15 / 15); preprocessing **fitted only on training data**.
- MLP with dropout, class-weighted binary cross entropy, early stopping and seeded experiments.
- ROC-AUC, PR-AUC, F1, precision, recall, accuracy measured **on held-out test data**.
- Saved model (`artifacts/model.pt`), reproducible preprocessing (`pipeline.joblib`) and a Streamlit dashboard for batch scoring.

## Data ethics and limitations

**The runnable local demonstration trains on simulated data, not real telecom customer records.** Its reported metrics are *demonstration-only* and do not validate real-world predictive performance, fairness, or causal impacts on retention. It is not suitable for automated interventions or production deployment. The public IBM Telco Customer Churn sample is also a fictional telecom-company example, not actual Orange Maroc data.

To switch to a public dataset, obtain [IBM's Telco Customer Churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) (7,043 rows / 21 fields) and run `python train.py --csv path/to/Telco-Customer-Churn.csv`. Data is not redistributed here.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Use `python train.py --csv path/to/IBM-Telco-Customer-Churn.csv` to retrain on the IBM sample. The generated `artifacts/` directory contains the fitted pipeline and evaluation results. Only load artifacts you created yourself: serialized preprocessing includes joblib (pickle).

## Business value

Demonstrates the workflow from customer attributes to segmented churn risk and an analytics dashboard for a marketing/retention team. The next step toward realistic telecom use is temporal validation on historical subscriber events, drift monitoring, model calibration, feature importance and threshold tuning based on campaign capacity and customer value.
