from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid


# request body for model
class ModelInput(BaseModel):
    model_name: str
    input_string: str  | None = None

app = FastAPI()


# get all the models 
@app.get("/api/v1/models")
async def get_all_models():
    #TODO Get all the models and return 
    return {"message": "TODO get all models"}


#post a phrase and train the model
@app.post("/api/v1/model/train")
async def train_model(input_data: ModelInput):

    # Extract the model name and input string from the request body
    model_name = input_data.model_name
    input_string = input_data.input_string


    #TODO train the model wiht the fraze, retrun confirmation when done

    #return status message
    return {"message": "TODO train model"}




#post an image and have the model classiy it
@app.post("/api/v1/model/classify")
async def classify_image(model_input: ModelInput, file: UploadFile = File(...)):


    file.filename = f"{uuid.uuid4()}.jpg"

    contents = await file.read()

    #TODO

    #return name of image
    return {"filename": file.filename}



#generate llm answers
@app.post("/api/v1/llm/generate")
async def llm_generate(input_data: ModelInput):
    # Extract the model name and input string from the request body
    model_name = input_data.model_name
    input_string = input_data.input_string
    #TODO train the model wiht the fraze, retrun confirmation when done

    #return status message
    return {"message": "TODO train model"}








#generate llm embeding of image
@app.post("/api/v1/llm/embedding")
async def llm_embeding(input_data: ModelInput):
    return {"message": "TODO train model"}