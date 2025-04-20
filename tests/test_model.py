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
        'LIMIT_BAL': 220000.0,
        'SEX': 1,
        'EDUCATION': 1,
        'MARRIAGE': 2,
        'AGE': 29,
        'PAY_0': 1,
        'PAY_2': 2,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 31012.0,
        'BILL_AMT2': 30215.0,
        'BILL_AMT3': 33117.0,
        'BILL_AMT4': 32286.0,
        'BILL_AMT5': 34320.0,
        'BILL_AMT6': 33634.0,
        'PAY_AMT1': 0.0,
        'PAY_AMT2': 3400.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 2576.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 2671.0
    }])
    print(model.metadata)
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_out_0(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 200000.0,
        'SEX': 1,
        'EDUCATION': 1,
        'MARRIAGE': 1,
        'AGE': 29,
        'PAY_0': -1,
        'PAY_2': -1,
        'PAY_3': -1,
        'PAY_4': -1,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 9585.0,
        'BILL_AMT2': 1686.0,
        'BILL_AMT3': 820.0,
        'BILL_AMT4': 1889.0,
        'BILL_AMT5': 6617.0,
        'BILL_AMT6': 3986.0,
        'PAY_AMT1': 1735.0,
        'PAY_AMT2': 824.0,
        'PAY_AMT3': 1889.0,
        'PAY_AMT4': 6617.0,
        'PAY_AMT5': 3986.0,
        'PAY_AMT6': 2460.0
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

