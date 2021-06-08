import os
import cv2
import regex as re
import pandas as pd
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path, height=200, width=200):

    filenames = [f for f in os.listdir(folder_path) if not f.startswith('.')]
    filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    images = []

    for filename in filenames:
        img_cv = cv2.imread(os.path.join(folder_path, filename))
        if img_cv is not None:
            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (height, width))
            images.append(img)
    return images


def load_image_data_into_dataframe(folder_path):
    filenames = [f for f in os.listdir(folder_path)]
    filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    df = pd.DataFrame()

    for filename in filenames:
        parsed_name = filename.split('_')
        df = df.append({'age': int(parsed_name[0])
                        # ,'gender' : parsed_name[1],
                        # 'race' : parsed_name[2]
                        }, ignore_index=True)

    # df['gender'] = df['gender'].map({'0': 'male', '1': 'female'})
    # df['race'] = df['race'].map({'0': 'White', '1': 'Black',
    # '2': 'Asian', '3': 'Indian', '4': 'Others'})

    return df


def show_images(img_list, img_df):
    n = int(len(img_list)/5) + 1
    f = plt.figure(figsize=(25, n*5))

    for i, img in enumerate(img_list):
        f.add_subplot(n, 5, i + 1)
        plt.title(f'Age: {img_df.iloc[i]["age"]}; \
                    Race: {img_df.iloc[i]["race"]} Index: {i}',
                  fontweight="bold", fontsize=15)
        plt.imshow(img)


def plot_history(history, title='', axs=None, exp_name=""):
    plt.figure(figsize=(15, 5))
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy' + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy' + exp_name)
    ax2.set_ylim(0.25, 0.5)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)


def convert_number(num):
    return f"{num*5+1}-{num*5+5}"
