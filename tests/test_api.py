from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "AverageMonthlySpend": 70.0,
        "IsNewCustomer": 0,
        "HasLongTermContract": 0
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result

    assert result["prediction"] in ["Churn", "No Churn"]

    assert 0 <= result["churn_probability"] <= 1


def test_explain():

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "AverageMonthlySpend": 70.0,
        "IsNewCustomer": 0,
        "HasLongTermContract": 0
    }

    response = client.post("/explain", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result
    assert "top_features" in result

    assert result["prediction"] in ["Churn", "No Churn"]
    assert 0 <= result["churn_probability"] <= 1
    assert len(result["top_features"]) > 0