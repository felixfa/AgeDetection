from preprocessing import target_binning, split_data
from utils import load_images_from_folder, load_image_data_into_dataframe, plot_history
from metrics import one_off_accuracy
from utils_CNN import CNN_fit

# Loading the Images 
X = load_images_from_folder(folder_path, height=100, width=100)
print("Images Loaded")

# Loading the target
targets = load_image_data_into_dataframe(folder_path)
print("Dataframe Loaded")

# Binning the y
y = target_binning(target)

# Train Test Split and Preprocessing
X_train,X_test,y_train,y_test,y_train_cat,y_test_cat = split_data(X,y,test_size=0.2,random_state=42,clear_mem=True)

# Load CNN
model = initialize_compile_model()

# Fit CNN
history = CNN_fit(model,patience=5)

# Predictions
y_pred = model.predict(X_test)

# 1-Off Accuracy Matrixes
matrix,conf = one_off_accuracy(y_test_cat,y_pred)
