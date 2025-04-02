import json
import pytest
import pandas as pd
import mlflow
from pathlib import Path


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    #mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    uri = "../../mlruns"
    Path(uri).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{

        'LIMIT_BAL': 0,
        'SEX': 1,
        'EDUCATION':1,
        'MARRIAGE': 1,
        'AGE': 31,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 0,
        'BILL_AMT2': 0,
        'BILL_AMT3': 0,
        'BILL_AMT4': 0,
        'BILL_AMT5': 0,
        'BILL_AMT6': 0,
        'PAY_AMT1': 0,
        'PAY_AMT2': 0,
        'PAY_AMT3': 0,
        'PAY_AMT4': 0,
        'PAY_AMT5': 0,
        'PAY_AMT6': 0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{

        'LIMIT_BAL': 0,
        'SEX': 1,
        'EDUCATION':1,
        'MARRIAGE': 1,
        'AGE': 31,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 0,
        'BILL_AMT2': 0,
        'BILL_AMT3': 0,
        'BILL_AMT4': 0,
        'BILL_AMT5': 0,
        'BILL_AMT6': 0,
        'PAY_AMT1': 0,
        'PAY_AMT2': 0,
        'PAY_AMT3': 0,
        'PAY_AMT4': 0,
        'PAY_AMT5': 0,
        'PAY_AMT6': 0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1


# def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
#     input = pd.DataFrame.from_records([{
#         'Pregnancies': 0,
#         'Glucose': 30,
#         'BloodPressure': 88,
#         'SkinThickness': 60,
#         'Insulin': 110,
#         'BMI': 20.0,
#         'DiabetesPedigreeFunction': 0.962,
#         'Age': 20
#     }])
#     prediction = model.predict(data=input)
#     assert prediction.shape == (1, )

# uri = "../../mlruns"
# Path(uri).mkdir(parents=True, exist_ok=True)
# mlflow.set_tracking_uri(uri)

