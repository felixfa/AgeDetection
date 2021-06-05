from tensorflow.keras import models
from os import path
from utils import load_images_from_folder, convert_number
import numpy as np
from PIL import Image
from autocrop import Cropper
import cv2


def loading_the_model():
    # load the model from disk
    if path.exists('../models/best_model2') == False:
        print('TO DO')
        #TO DO
    print('Nothing to Download!')
    model = models.load_model('../models/best_model2')
    return model

def loading_one_image():
    cropper = Cropper(width=100,height=100)

    # Loading one picture
    #cropped_array = cropper.crop('../raw_data/Test/IMG_2278.JPG') #Lesly
    #cropped_array = cropper.crop('../raw_data/Test/IMG_2278.JPG') #Felix
    cropped_array = cropper.crop('../raw_data/Test/0d33a016-7cdb-4184-938a-ffae451a7eda.JPG') #Rami
    #cropped_array = cropper.crop('../raw_data/Test/Bildschirmfoto 2021-06-05 um 16.30.02.png') #Tiago

    cropped_array = np.expand_dims(cropped_array,axis=0)
    return cropped_array

def predict(model,X):
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))
    return y_pred

if __name__ == '__main__':
    model = loading_the_model()
    X = loading_one_image()
    y_pred = predict(model,X)
    print(f'Prediction: {y_pred}')



