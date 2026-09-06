from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "Laybaatariq/fraud-detection-model"

MODEL_FILENAME = "fraud_model_paysim_baseline.pkl"
SCALER_FILENAME = "scaler_paysim.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns_paysim.pkl"
EXPLAINER_FILENAME = "shap_explainer_paysim.pkl"


def load_artifacts():
    print("Loading model artifacts from Hugging Face Hub...")

    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename=SCALER_FILENAME)
    feature_columns_path = hf_hub_download(repo_id=REPO_ID, filename=FEATURE_COLUMNS_FILENAME)
    explainer_path = hf_hub_download(repo_id=REPO_ID, filename=EXPLAINER_FILENAME)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    explainer = joblib.load(explainer_path)

    print("Model artifacts loaded successfully.")
    return model, scaler, feature_columns, explainer
