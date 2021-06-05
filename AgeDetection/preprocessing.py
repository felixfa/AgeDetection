from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def target_binning(targets, target='age'):

    bins = [5*i for i in range(17)]
    labels = np.linspace(0, 15, 16)
    y = pd.cut(targets[target], bins=bins, labels=labels)

    return y


def split_data(X, y, test_size=0.2, random_state=42, clear_mem=True):

    # Splitting
    X_train, X_test, y_train, y_test = \ 
    train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Adjusting data types
    X_train = np.array(X_train, dtype="float32")
    X_test =np.array(X_test, dtype="float32")
    y_train = np.array(y_train, dtype="int8")
    y_test =np.array(y_test, dtype="int8")

    # Scaling
    X_train = X_train/255 - 0.5
    X_test = X_test/255 - 0.5

    # Categorizing
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Deleting X,y
    if clear_mem:
        del X, y

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat