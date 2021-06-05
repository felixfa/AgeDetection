from numpy.random import seed
from tensorflow.random import set_seed

from tensorflow.keras.callbacks import EarlyStopping


def CNN_fit(model,X_train,y_train_cat,epochs=100,batch_size=16,patience=5):
    seed(1)
    set_seed(1)

    es = EarlyStopping(patience=patience,restore_best_weights=True)

    history = model.fit(X_train, y_train_cat, 
                                    validation_split=0.2,
                                    callbacks=[es],
                                    epochs=epochs, 
                                    batch_size=batch_size
                                    )

    #print(f'Score: {model.evaluate(X_test, y_test_cat, verbose=0)[1]}')

    return history