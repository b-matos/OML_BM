import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, conint, confloat, Field
from typing import Annotated
import pandas as pd
import json
import uvicorn

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
    LIMIT_BAL: confloat(ge=0) = 0
    SEX: int = 1
    EDUCATION: int = 1
    MARRIAGE: int = 1
    AGE: int = 31
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = 0
    PAY_AMT2: float = 0
    PAY_AMT3: float = 0
    PAY_AMT4: float = 0
    PAY_AMT5: float = 0
    PAY_AMT6: float = 0


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
