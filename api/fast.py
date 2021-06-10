from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
model_path = '/home/fruntxas/code/felixfa/AgeDetection/models/best_model'

model = models.load_model(model_path)


def predict(model, X):
    y_pred = model.predict(X)
    return y_pred


def load_image_into_numpy_array(image):
    cropper = Cropper(width=100, height=100)
    cropped_array = cropper.crop(image)
    return np.expand_dims(cropped_array, axis=0)


def convert_number(num):
    return f"{num*5+1}-{num*5+5}"


# Upload Image
@app.post("/image")
async def read_root(file: UploadFile = File(...)):

    with open("tmp.png", "wb+") as data:
        data.write(file.file.read())
        print("Written to image")

    X = load_image_into_numpy_array("tmp.png")
    X = X/255 - 0.5
    print("Image converted to array")
    y_pred = predict(model, X)
    print(y_pred)
    print(type(y_pred))
    guess = convert_number(int(np.argsort(y_pred[0])[-1]))
    print("Guess Performed")
    return {"Guess": guess}


@app.get("/")
def index():
    return {"Greetings": "Welcome to the Age Prediction"}
