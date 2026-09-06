# fraud-detection-project

A full-stack fraud detection system built on 6.3M+ mobile money transactions (PaySim dataset), combining a tuned XGBoost classifier with SHAP-based explainability, served through a FastAPI backend and a Streamlit dashboard.

## Overview

Financial fraud detection faces a core challenge: fraud is extremely rare (~0.13% of transactions here), so naive accuracy is meaningless, and a black-box prediction alone isn't actionable for a fraud analyst. This project addresses both problems:

- A classifier trained and evaluated with metrics suited to severe class imbalance (precision, recall, PR-AUC — not accuracy)
- A SHAP explainability layer that turns every prediction into a plain-language explanation of the contributing factors
- A live dashboard where a user can enter transaction details and get an instant, explained fraud score

## Screenshot

![Fraud checker dashboard](docs/screenshot-frontend.png)

## Architecture

Streamlit (frontend) -> FastAPI (backend) -> XGBoost model + SHAP explainer
|
Model artifacts hosted on Hugging Face Hub


## Dataset

**PaySim Mobile Money Fraud Dataset** (Kaggle: `ealaxi/paysim1`) - 6.3M simulated mobile money transactions, ~0.13% labeled fraud.

Key EDA findings:
- Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types
- Fraud transactions typically involve larger amounts and a distinct sender-balance pattern (full account drain)
- Raw `amount` and balance columns are heavily right-skewed (skew 10-20), requiring log transformation

## Feature engineering

| Feature | Purpose |
|---|---|
| `amount_log`, `*_balance_log` | Log-transformed to correct heavy right-skew |
| `orig_balance_was_zero`, `dest_balance_was_zero` | Captures the zero-balance pattern discovered in EDA |
| `type_*` (one-hot) | Encodes transaction type, since fraud concentrates in specific types |
| `hour_of_day` | Extracted from the `step` column to capture time-of-day patterns |

**Note:** An initial `balance_diff_orig`/`balance_diff_dest` feature (sender/receiver balance discrepancy) was engineered but discarded - it was so strongly correlated with the fraud label that the model achieved a suspicious 100% accuracy, indicating data leakage rather than genuine learning. Removing it produced a more realistic, trustworthy model.

## Model

- **Algorithm:** XGBoost, with `scale_pos_weight` to handle class imbalance
- **Hyperparameter tuning** was evaluated via `RandomizedSearchCV` but showed no meaningful improvement over sensible defaults, so the simpler baseline model was kept for production
- **Validated with 5-fold stratified cross-validation** to confirm stability across data splits

### Final performance (held-out test set)

| Metric | Legit | Fraud |
|---|---|---|
| Precision | 1.00 | 0.71 |
| Recall | 1.00 | 0.96 |
| F1-score | 1.00 | 0.82 |

**ROC-AUC:** 0.9998 - **PR-AUC:** 0.9684

## Explainability

Every prediction is paired with a SHAP explanation identifying the top contributing features (e.g. "Flagged due to a TRANSFER emptying the sender's balance to near zero, combined with an unusually large amount"), making the system usable by a human fraud analyst rather than a black box.

## Tech stack

- **Model training:** Python, scikit-learn, XGBoost, SHAP (Google Colab)
- **Model hosting:** Hugging Face Hub
- **Backend:** FastAPI
- **Frontend:** Streamlit, Plotly
- **Dev environment:** GitHub Codespaces

## Project structure

├── backend/
│ ├── main.py # FastAPI app - /predict, /explain endpoints
│ ├── model_loader.py # Downloads and loads model artifacts from Hugging Face
│ ├── preprocessing.py # Converts raw transaction input into model-ready features
│ └── requirements.txt
├── frontend/
│ ├── app.py # Streamlit dashboard
│ └── requirements.txt
├── notebooks/
│ └── train_model.ipynb # Full training pipeline: EDA, cleaning, training, evaluation
├── tests/
│ └── test_api.py # API endpoint tests
└── README.md


## Running locally

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend** (in a separate terminal):
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Key learnings

- Diagnosed and fixed a data leakage issue that was producing an artificially perfect model
- Learned that PR-AUC and threshold-specific metrics (precision/recall at 0.5) can diverge - a tuned model with better PR-AUC can still perform worse at the default threshold
- Validated that hyperparameter tuning doesn't always beat sensible defaults, rather than assuming it always helps
EOF