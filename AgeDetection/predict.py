from tensorflow.keras import models
from os import path
from utils import load_images_from_folder, convert_number
import numpy as np

BUCKET_NAME="age_detection_faehnrich"
STORAGE_LOCATION = 'models/best_model'

def loading_the_model():
    # load the model from disk
    if path.exists('../models/best_model') == False:
        print('TO DO')
        #TO DO
    print('Nothing to Download!')
    model = models.load_model('../models/best_model')
    return model

def loading_one_image():
    # One Sample image
    X = load_images_from_folder('../raw_data/Test', height=100, width=100)
    print("Image Loaded")
    return np.array(X)

def predict(model,X):
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))
    return y_pred

if __name__ == '__main__':
    model = loading_the_model()
    X = loading_one_image()
    y_pred = predict(model,X)
    print(f'Prediction: {y_pred}')



