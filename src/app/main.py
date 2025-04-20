import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, conint, confloat, Field
from enum import Enum
from typing import Annotated
import pandas as pd
import json
import uvicorn

class ValidInt(Enum):
    NEGATIVE_ONE = -1
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
        SEX: Gender (1=male, 2=female)
        EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
        MARRIAGE: Marital status (1=married, 2=single, 3=others)
        AGE: Age in years
        PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
        PAY_2: Repayment status in August, 2005 (scale same as above)
        PAY_3: Repayment status in July, 2005 (scale same as above)
        PAY_4: Repayment status in June, 2005 (scale same as above)
        PAY_5: Repayment status in May, 2005 (scale same as above)
        PAY_6: Repayment status in April, 2005 (scale same as above)
        BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
        BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
        BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
        BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
        BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
        BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
        PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
        PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
        PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
        PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
        PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
        PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
    """
    LIMIT_BAL: confloat(ge=0) = 220000.0
    SEX: conint(ge=1, le=2) = 1
    EDUCATION: conint(ge=1, le=6) = 1
    MARRIAGE: conint(ge=1 , le=3) = 2
    AGE: int = 29
    PAY_0: ValidInt = 1
    PAY_2: ValidInt = 2
    PAY_3: ValidInt = 2
    PAY_4: ValidInt = 2
    PAY_5: ValidInt = 2
    PAY_6: ValidInt = 2
    BILL_AMT1: confloat(ge=0) = 31012.0
    BILL_AMT2: confloat(ge=0) = 30215.0
    BILL_AMT3: confloat(ge=0) = 33117.0
    BILL_AMT4: confloat(ge=0) = 32286.0
    BILL_AMT5: confloat(ge=0) = 34320.0
    BILL_AMT6: confloat(ge=0) = 33634.0
    PAY_AMT1: confloat(ge=0) = 0.0
    PAY_AMT2: confloat(ge=0) = 3400.0
    PAY_AMT3: confloat(ge=0) = 0.0
    PAY_AMT4: confloat(ge=0) = 2576.0
    PAY_AMT5: confloat(ge=0) = 0.0
    PAY_AMT6: confloat(ge=0) = 2671.0


# Create a FastAPI application
app = fastapi.FastAPI()

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """

    MLFLOW_TRACKING_URI = f"{config['tracking_base_url']}:{config['tracking_port']}" # comment this line to run locally
    #MLFLOW_TRACKING_URI = f"http://localhost:{config['tracking_port']}" # uncomment this line to run locally
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 
    app.client = mlflow.tracking.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)
    print(f"Loaded model {model_uri}")


@app.post("/default_payment_prediction")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}

@app.get("/model_params")
async def get_params():
    """
    Endpoint to retrieve the parameters of the model.

    Returns:
        dict: A dictionary containing the model parameters.
    """
    RUN_ID = app.model.metadata.to_dict()['run_id']

    model_data = app.client.get_run(RUN_ID).data.to_dictionary()
    
    return model_data['params']

@app.get("/model_metrics")
async def get_metrics():
    """
    Endpoint to retrieve the metrics of the model.

    Returns:
        dict: A dictionary containing the model metrics.
    """
    RUN_ID = app.model.metadata.to_dict()['run_id']

    model_data = app.client.get_run(RUN_ID).data.to_dictionary()
    
    return model_data['metrics']

@app.get("/model_metadata")
async def get_metadata():
    """
    Endpoint to retrieve the metadata of the model.

    Returns:
        dict: A dictionary containing the model metadata.
    """

    return app.model.metadata.to_dict()

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the service is running.

    Returns:
        dict: A dictionary indicating that the service is healthy.
    """
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    try:
        mlflow.pyfunc.load_model(model_uri = model_uri)
        return {"status": "healthy"}
    except MlflowException as e:
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"MLflow client error: {e}"
        )

# Run the app on port 5002
uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
