from fastapi import FastAPI
import joblib
import pandas as pd
import shap
import numpy as np

app = FastAPI(
    title="Explainable Tabular ML Engine",
    version="1.0.0"
)

# Load model and preprocessing pipeline
model = joblib.load("models/random_forest_churn.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
explainer = shap.TreeExplainer(model)
feature_names = preprocessor.get_feature_names_out()

@app.get("/")
def home():
    return {
        "message": "Explainable Tabular ML Engine API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: dict):

    # Convert JSON to DataFrame
    data = pd.DataFrame([customer])

    # Apply same preprocessing used during training
    data_processed = preprocessor.transform(data)

    # Prediction
    prediction = model.predict(data_processed)[0]

    # Probability
    probability = model.predict_proba(data_processed)[0][1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }
@app.post("/explain")
def explain(customer: dict):

    # Convert input to DataFrame
    data = pd.DataFrame([customer])

    # Apply the same preprocessing
    data_processed = preprocessor.transform(data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(data_processed)

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        customer_shap = shap_values[1][0]

    elif len(shap_values.shape) == 3:
        customer_shap = shap_values[0, :, 1]

    else:
        customer_shap = shap_values[0]

    # Get top 10 important features
    importance = np.abs(customer_shap)

    top_indices = np.argsort(
        importance
    )[::-1][:10]

    explanations = []

    for i in top_indices:
        explanations.append({
            "feature": feature_names[i],
            "shap_value": round(float(customer_shap[i]), 4),
            "importance": round(float(importance[i]), 4)
        })

    prediction = model.predict(data_processed)[0]
    probability = model.predict_proba(data_processed)[0][1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4),
        "top_features": explanations
    }