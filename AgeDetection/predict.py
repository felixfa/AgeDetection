from tensorflow.keras import models
from utils import convert_number
import numpy as np
from autocrop import Cropper
# from PIL import Image
# import cv2


def loading_the_model():
    # load the model from disk
    model = models.load_model('../models/best_model')
    return model

def loading_images():
    cropper = Cropper(width=100,height=100)
    cropped_array = []
    # Loading images
    #cropped_array.append(cropper.crop('../test_data/IMG_2278.JPG')) #Lesly
    #cropped_array.append(cropper.crop('../test_data/IMG_2278.JPG')) #Felix
    #cropped_array.append(cropper.crop('../test_data/0d33a016-7cdb-4184-938a-ffae451a7eda.JPG')) #Rami
    #cropped_array.append(cropper.crop('../test_data/Bildschirmfoto 2021-06-05 um 16.30.02.png')) #Tiago
    cropped_array.append(cropper.crop('../test_data/Qiwei.png')) #Qiwei
    #cropped_array.append(cropper.crop('../test_data/Nicole.jpg')) #Nicole
    #cropped_array.append(cropper.crop('../test_data/perry.png')) #Matthew Perry
    cropped_array = np.array(cropped_array)
    cropped_array = cropped_array / 255 - 0.5
    #cropped_array = np.expand_dims(cropped_array,axis=0)
    return cropped_array


def predict(model,X):
    y_pred = model.predict(X)
    return y_pred


if __name__ == '__main__':
    model = loading_the_model()
    X = loading_images()
    y_pred = predict(model,X)
    print(f'Prediction_1: {convert_number(int(np.argsort(y_pred[0])[-1]))}')
    print(f'Age Bin_1:  {int(np.argsort(y_pred[0])[-1])}')
    print(f'Prediction_2: {convert_number(int(np.argsort(y_pred[0])[-2]))}')
    print(f'Age Bin_2:  {int(np.argsort(y_pred[0])[-2])}')
    print(f'Prediction_3: {convert_number(int(np.argsort(y_pred[0])[-3]))}')
    print(f'Age Bin_3:  {int(np.argsort(y_pred[0])[-3])}')
    print(f'Propability: {y_pred}')



