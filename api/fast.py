from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models
import numpy as np
from api.utils_basic import image_to_array, convert_weight, age_range, weighted_accuracy, predict

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_path = 'models/best_model'
model = models.load_model(model_path)


# Upload Image
@app.post("/image")
async def read_root(file: UploadFile = File(...)):

    with open("tmp.png", "wb+") as data:
        data.write(file.file.read())
        print("Written to image")

    # Converting Image to Array and Scaling
    X = image_to_array("tmp.png")
    X = X/255 - 0.5
    print("Scaled Image converted to array")

    # Predicting
    y_pred = predict(model, X)  # [0]
    print(y_pred)
    print("Prediction Performed")

    #Main Bin
    main_pred = np.argmax(y_pred)

    # Pred List for weighted prediction
    print(main_pred)
    weighted_bin, weighted_pred = weighted_accuracy(y_pred)
    print(weighted_bin)

    # guess = round(modf(weighted_pred)[1]*5+1 + modf(weighted_pred)[0] * 5, 2)

    output = {"Initial Age Bin": age_range(weighted_bin), "Weighted Guess": weighted_pred}
    print("Guesses Performed")

    return output


@app.get("/")
def index():
    return {"Greetings": "Welcome to the Age Prediction"}

# main_pred = np.argmax(y_pred)
    '''pred_bins = {"Main Guess" :
        {"Range": convert_number(main_pred), "Probability": float(y_pred[main_pred])},
                 "Left Guess" :
        {"Range": convert_number(main_pred-1), "Probability": float(y_pred[main_pred-1])},
                 "Right Guess":
        {"Range": convert_number(main_pred+1), "Probability": float(y_pred[main_pred+1])}
    }'''
