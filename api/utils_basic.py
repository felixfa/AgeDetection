from autocrop import Cropper
import numpy as np
from tensorflow.keras import models

def age_range(num):
    return f"{num*5+1-5}-{num*5+5+5}"


def convert_weight(num):
    return f"{num*5+1}"


def image_to_array(image):
    cropper = Cropper(width=100, height=100)
    cropped_array = cropper.crop(image)
    return np.expand_dims(cropped_array, axis=0)

def weighted_accuracy(y_pred):
    weighted_pred = 0
    for i, element in enumerate(y_pred):
        num_1 = int(np.argsort(y_pred[i])[-1])
        num_2 = int(np.argsort(y_pred[i])[-2])
        num_3 = int(np.argsort(y_pred[i])[-3])
        prop_1 = y_pred[i][num_1]
        prop_2 = y_pred[i][num_2]
        prop_3 = y_pred[i][num_3]
        prop_sum = prop_1+prop_2+prop_3
        pred = num_1*prop_1/prop_sum+num_2*prop_2/prop_sum+num_3*prop_3/prop_sum
    return pred


def predict(model, X):
    y_pred = model.predict(X)
    return y_pred
