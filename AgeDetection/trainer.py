from AgeDetection.preprocessing import target_binning, split_data
from AgeDetection.utils import load_images_from_folder, load_image_data_into_dataframe, plot_history
from AgeDetection.metrics import one_off_accuracy
from AgeDetection.utils_CNN import CNN_fit
from AgeDetection.CNN import initialize_compile_model

folder_path = "/home/fruntxas/code/felixfa/AgeDetection/raw_data/UTKFace"     

if __name__ == "__main__":
    # Loading the Images 
    print("Starting name/main\n")
    X = load_images_from_folder(folder_path, height=100, width=100)
    print("Images Loaded\n")

    # Loading the target
    targets = load_image_data_into_dataframe(folder_path)
    print("Dataframe Loaded\n")

    # Binning the y
    y = target_binning(targets,target='age')
    print("Target Set\n")

    # Train Test Split and Preprocessing
    X_train,X_test,y_train,y_test,y_train_cat,y_test_cat = split_data(X,y,test_size=0.2,random_state=42,clear_mem=True)
    print("Split Done\n")

    # Load CNN
    model = initialize_compile_model(input_shape=(100,100,3),categories=16)
    print("Model Initialized\n")

    # Fit CNN
    history = CNN_fit(model,X_train,y_train_cat,epochs=1,batch_size=16,patience=5)
    print("Model Fitted\n")
    print(f'Model Evaluate Score: {model.evaluate(X_test, y_test_cat, verbose=0)[1]}')

    # Predictions
    y_pred = model.predict(X_test)
    print("y_pred created")

    # 1-Off Accuracy Matrixes
    matrix,conf = one_off_accuracy(y_test_cat,y_pred)

    print(conf)
