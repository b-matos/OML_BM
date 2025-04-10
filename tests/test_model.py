import json
import pytest
import pandas as pd
import mlflow
import mlflow.pyfunc
from pathlib import Path


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    #MLFLOW_TRACKING_URI = f"{config['tracking_base_url']}:{config['tracking_port']}"
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}") #uncomment this line to run locally
    #mlflow.set_registry_uri(MLFLOW_TRACKING_URI) # comment this line to run locally
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out_1(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
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
    }])
    print(model.metadata)
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_out_0(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 200000.0,
        'SEX': 2.0,
        'EDUCATION': 2.0,
        'MARRIAGE': 1.0,
        'AGE': 50.0,
        'PAY_0': -2.0,
        'PAY_2': -2.0,
        'PAY_3': -2.0,
        'PAY_4': -2.0,
        'PAY_5': -2.0,
        'PAY_6': -2.0,
        'BILL_AMT1': 411.0,
        'BILL_AMT2': 453.0,
        'BILL_AMT3': 348.0,
        'BILL_AMT4': 359.0,
        'BILL_AMT5': 672.0,
        'BILL_AMT6': 1620.0,
        'PAY_AMT1': 453.0,
        'PAY_AMT2': 348.0,
        'PAY_AMT3': 359.0,
        'PAY_AMT4': 672.0,
        'PAY_AMT5': 1620.0,
        'PAY_AMT6': 384.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 50000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 36,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 2,
        'BILL_AMT1': 94228.0,
        'BILL_AMT2': 47635.0,
        'BILL_AMT3': 42361.0,
        'BILL_AMT4': 19574.0,
        'BILL_AMT5': 20295.0,
        'BILL_AMT6': 19439.0,
        'PAY_AMT1': 2000.0,
        'PAY_AMT2': 1500.0,
        'PAY_AMT3': 1000.0,
        'PAY_AMT4': 1800.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 1000.0
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )

