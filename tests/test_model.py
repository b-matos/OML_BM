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
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
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


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 2,
        'AGE': 25,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 8864.0,
        'BILL_AMT2': 10062.0,
        'BILL_AMT3': 11581.0,
        'BILL_AMT4': 12580.0,
        'BILL_AMT5': 13716.0,
        'BILL_AMT6': 14828.0,
        'PAY_AMT1': 1500.0,
        'PAY_AMT2': 2000.0,
        'PAY_AMT3': 1500.0,
        'PAY_AMT4': 1500.0,
        'PAY_AMT5': 1500.0,
        'PAY_AMT6': 2000.0
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

