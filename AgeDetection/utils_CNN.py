from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow import random

from tensorflow.keras.callbacks import EarlyStopping


def CNN_fit(model,patience=5):
    seed(1)
    random.set_seed(1)

    es = EarlyStopping(patience=patience,restore_best_weights=True)

    history = model.fit(X_train, y_train_cat, 
                                    validation_split=0.2,
                                    callbacks=[es],
                                    epochs=100, 
                                    batch_size=16
                                    )

    print(f'Score: {model.evaluate(X_test, y_test_cat, verbose=0)[1]}')

    return history