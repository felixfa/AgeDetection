from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Greetings": "Welcome to the Age Prediction"}

@app.get("/predict_age")
def predict_fare(image):

    #url = 'http://localhost:8000/predict_fare'
    
    # Image to Array
    cropper = Cropper(width=100,height=100)
    cropped_array = cropper.crop(image)
    
    X = np.expand_dims(cropped_array,axis=0)
    
    # Load Model Locally
    model = models.load_model('/home/fruntxas/code/felixfa/AgeDetection/models/best_model2')

    # Predict
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))

    return y_pred

    # Load Model from GCP
    # from google.cloud import storage

    # BUCKET_NAME = 'wagon-ml-pereira-566'
    # STORAGE_LOCATION = 'model.joblib'

    # client = storage.Client()
    # bucket = client.get_bucket(BUCKET_NAME)
    # blob = bucket.blob(STORAGE_LOCATION)

    # blob.download_to_filename('teste.joblib')

    #=> {wait: 64}
