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

    X_pred = pd.DataFrame([params.values()],columns=params.keys())

    # Load Model Locally
    model = models.load_model('/home/fruntxas/code/felixfa/AgeDetection/models/best_model2')

    # Load Model from GCP
    # from google.cloud import storage

    # BUCKET_NAME = 'wagon-ml-pereira-566'
    # STORAGE_LOCATION = 'model.joblib'

    # client = storage.Client()
    # bucket = client.get_bucket(BUCKET_NAME)
    # blob = bucket.blob(STORAGE_LOCATION)

    # blob.download_to_filename('teste.joblib')

    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))

    return y_pred
    #=> {wait: 64}
