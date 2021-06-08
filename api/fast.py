from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from autocrop import Cropper
from tensorflow.keras import models
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = models.load_model('/home/fruntxas/code/felixfa/AgeDetection/models/best_model2')

cropper = Cropper(width=100,height=100)

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/image")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    #print(image.data)
    return image#{"Hello": "World"}


@app.post("/try")
def predict(data_diabetes:float):
   data = np.array([[data_diabetes]])
   prediction = model.predict(data)
   return {
       'prediction': prediction[0],
   }


@app.get("/")
def index():
    return {"Greetings": "Welcome to the Age Prediction"}


#@app.post("/uploadfile/")
#async def create_upload_file(file: UploadFile = File(...)):
#    return {"filename": file.filename}


@app.post("/predict_age")
def predict_fare(image_array):

    # Image to Array
    cropped_array = cropper.crop(image_array)
    X = np.expand_dims(cropped_array,axis=0)

    # Predict
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))
 
    return {'prediction' : y_pred}
    #return image_array

