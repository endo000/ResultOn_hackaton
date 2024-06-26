from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, WebSocket
from pydantic import BaseModel
import os
from mlflow import MlflowClient
from dotenv import load_dotenv
from utils.execute_ssh import execute_ssh_command
from utils.s3_upload_file import s3_upload_file
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from fastapi.responses import HTMLResponse
from utils.rabbitmq import RabbitMQ


# request body for model
class ModelInput(BaseModel):
    model_name: str
    input_string: str  | None = None


# method to make enviroment variables avalable
load_dotenv()

# startup the FastAPI app
app = FastAPI()


# get all models
@app.get("/api/v1/models")
async def get_all_models():

    try:
        client = MlflowClient()
        list_of_models = [dict(rm) for rm in client.search_model_versions()]

        return {"models": list_of_models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to connect to MLflow server")




#post a phrase and train the model
@app.post("/api/v1/model/train")
async def train_model(input_data: ModelInput):

    # Extract the model name and input string from the request body
    model_name = input_data.model_name
    input_string = input_data.input_string

    # execute command via custom method and return to api the anwser from the ssh (or error)
    output = execute_ssh_command(os.environ["COMMAND_TRAIN"])


    

    if output is not None:
        print("Command output:")
        print(output)
        return {"message": output}
    else:
        return {"message": "An error has occured"}



#post an image and have the model classiy it
@app.post("/api/v1/model/classify")
async def classify_image(model_input: ModelInput = Depends(), file: UploadFile = File(...)):

    # upload image
    filename = s3_upload_file(file)

    # execute command via custom method and return to api the anwser from the ssh (or error)
    output = execute_ssh_command('bash hackathon/hpc/bin/mobilenetv3.sh')

    if output is not None:
        print("Command output:")
        print(output)
        return {"message": output}
    else:
        return {"message": "An error has occured"}



#generate llm answers
@app.post("/api/v1/llm/generate")
async def llm_generate(prompt = str):
    print(f'bash hackathon/hpc/bin/alpaca_llm.sh {prompt}')
    # execute command via custom method and return to api the anwser from the ssh (or error)
    output = execute_ssh_command(f'bash alpaca_llm.sh {prompt}')

#def callback
    #rabbit = RabbitMQ(routing_key="#")
    #str = rabbit.receive()

    

    if output is not None:
        print("Command output:")
        print(output)
        return {"message": output}
    else:
        return {"message": "An error has occured"}









#generate llm embeding of image
@app.post("/api/v1/llm/embedding")
async def llm_embeding(file: UploadFile = File(...)):

    # upload image
    filename = s3_upload_file(file)

    # execute command via custom method and return to api the anwser from the ssh (or error)
    output = execute_ssh_command('bash hackathon/hpc/bin/mobilenetv3.sh')
    if output is not None:
        print("Command output:")
        print(output)
        return {"message": output}
    else:
        return {"message": "An error has occured"}
