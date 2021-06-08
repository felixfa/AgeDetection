from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import joblib
from autocrop import Cropper
from tensorflow.keras import models
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
model2_path = '/home/fruntxas/code/felixfa/AgeDetection/models/best_model2'
model = models.load_model(model2_path)



def load_image_into_numpy_array(image):
    cropper = Cropper(width=100, height=100)
    cropped_array = cropper.crop(image)
    return np.expand_dims(cropped_array, axis=0)


def convert_number(num):
    return f"{num*5+1}-{num*5+5}"


# Upload Image
@app.post("/image")
async def read_root(file: UploadFile = File(...)):
    print(type(file))
    print(file)
    # print(file.content)
    #image = load_image_into_numpy_array(await file.read())
    # print(image.data)
    return file  # {"Hello": "World"}


@app.get("/")
def index():
    return {"Greetings": "Welcome to the Age Prediction"}



# Not like this I think
@app.post("/predict_age")
def predict_fare(image_array):

    # Image to Array
    cropped_array = cropper.crop(image_array)
    X = np.expand_dims(cropped_array, axis=0)

    # Predict
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))

    # return {'prediction': y_pred}
    return image_array
