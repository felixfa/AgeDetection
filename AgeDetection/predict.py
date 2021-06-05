from tensorflow.keras import models
from os import path
from AgeDetection.utils import convert_number  # , load_images_from_folder
import numpy as np
from autocrop import Cropper
# from PIL import Image
# import cv2


def loading_the_model():
    # load the model from disk
    if not path.exists('../models/best_model2'):
        print('TO DO')
        # TO DO
    print('Nothing to Download!')
    model2_path = '/home/fruntxas/code/felixfa/AgeDetection/models/best_model2'
    model = models.load_model(model2_path)
    return model


def loading_one_image():
    cropper = Cropper(width=100, height=100)
    test_path = "/home/fruntxas/code/felixfa/AgeDetection/test_data/"
    # Loading one picture
    # cropped_array = cropper.crop(test_path+'IMG_2278.JPG') # Lesly
    # cropped_array = cropper.crop(test_path+'IMG_2278.JPG') # Felix
    # cropped_array = cropper.crop(test_path+'Rami.JPG') # Rami
    cropped_array = cropper.crop(test_path+'Tiago.png')  # Tiago

    cropped_array = np.expand_dims(cropped_array, axis=0)
    return cropped_array


def predict(model, X):
    y_pred = model.predict(X)
    y_pred = convert_number(int(np.argsort(y_pred[0])[-1]))
    return y_pred


if __name__ == '__main__':
    model = loading_the_model()
    X = loading_one_image()
    y_pred = predict(model, X)
    print(f'Prediction: {y_pred}')
