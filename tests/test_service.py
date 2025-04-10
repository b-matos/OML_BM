import json
import pytest
import requests

with open('./config/app.json') as f:
    config = json.load(f)

def test_default_payment_prediction():
    """
    Test for the /default_payment endpoint with valid input data.
    It should return a prediction in the response.
    """

    response = requests.post(f"http://localhost:{config["service_port"]}/default_payment_prediction", json={ # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/default_payment_prediction", json={ # comment this line to run locally



        'LIMIT_BAL': 30000.0,
        'SEX': 2,
        'EDUCATION': 3,
        'MARRIAGE': 1,
        'AGE': 58,
        'PAY_0': 2,
        'PAY_2': 2,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 26580.0,
        'BILL_AMT2': 27652.0,
        'BILL_AMT3': 27905.0,
        'BILL_AMT4': 27158.0,
        'BILL_AMT5': 29239.0,
        'BILL_AMT6': 29791.0,
        'PAY_AMT1': 1800.0,
        'PAY_AMT2': 1000.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 2518.0,
        'PAY_AMT5': 1185.0,
        'PAY_AMT6': 1200.0
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))
    assert response.json()["prediction"] == 1

    # Criar testes para gets

def test_default_payment_prediction_invalid_data():
    """
    Test for the /default_payment endpoint with invalid input data.
    It should return a 422 Unprocessable Entity status code.
    """

    response = requests.post(f"http://localhost:{config["service_port"]}/default_payment_prediction", json={ # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/default_payment_prediction", json={ # comment this line to run locally

        'LIMIT_BAL': 30000.0,
        'SEX': 2,
        'EDUCATION': 3.5, # Invalid value
        'MARRIAGE': 1,
        'AGE': 58,
        'PAY_0': 2,
        'PAY_2': 2,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 26580.0,
        'BILL_AMT2': 27652.0,
        'BILL_AMT3': 27905.0,
        'BILL_AMT4': 27158.0,
        'BILL_AMT5': 29239.0,
        'BILL_AMT6': 29791.0,
        'PAY_AMT1': 1800.0,
        'PAY_AMT2': 1000.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 2518.0,
        'PAY_AMT5': 1185.0,
        'PAY_AMT6': 1200.0
    })
    assert response.status_code == 422

def test_default_payment_prediction_missing_data():
    """
    Test for the /default_payment endpoint with missing input data.
    It should return a 422 Unprocessable Entity status code.
    """
    response = requests.post(f"http://localhost:{config["service_port"]}/default_payment_prediction") # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/default_payment_prediction") # comment this line to run locally

    assert response.status_code == 422    


# Test get endpoints
def test_get_model_metrics():
    """
    Test for the /model endpoint.
    It should return the model metrics.
    """

    response = requests.get(f"http://localhost:{config["service_port"]}/model_metrics") # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/model_metrics") # comment this line to run locally
    assert response.status_code == 200
    assert len(response.json()) == 6

def test_get_model_params():
    """
    Test for the /model endpoint.
    It should return the model params.
    """
    response = requests.get(f"http://localhost:{config["service_port"]}/model_params") # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/model_params") # comment this line to run locally
    assert response.status_code == 200

def test_get_model_metadata():
    """
    Test for the /model endpoint.
    It should return the model metadata.
    """
    response = requests.get(f"http://localhost:{config["service_port"]}/model_metadata") # uncomment this line to run locally
    #response = requests.post(f"{config['service_base_url']}:{config['service_port']}/model_metadata") # comment this line to run locally
    assert response.status_code == 200