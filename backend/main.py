from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_loader import load_artifacts
from preprocessing import preprocess_transaction

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, scaler, feature_columns, explainer = load_artifacts()


class Transaction(BaseModel):
    transaction_type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    hour_of_day: int
    step: int = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_df = preprocess_transaction(transaction, scaler, feature_columns)
        fraud_prob = float(model.predict_proba(input_df)[0][1])
        prediction = "Fraud" if fraud_prob >= 0.5 else "Legit"
        return {"fraud_probability": round(fraud_prob, 4), "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain")
def explain(transaction: Transaction):
    try:
        input_df = preprocess_transaction(transaction, scaler, feature_columns)
        shap_values = explainer.shap_values(input_df)

        feature_shap_pairs = list(zip(input_df.columns.tolist(), shap_values[0]))
        feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_shap_pairs[:4]

        feature_desc = ", ".join(
            [f"{name} ({float(value):+.2f})" for name, value in top_features]
        )
        explanation = f"Top contributing factors: {feature_desc}."

        return {
            "explanation": explanation,
            "top_features": [{"feature": n, "shap_value": round(float(v), 4)} for n, v in top_features]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
