from google.cloud import storage
from zipfile import ZipFile
from os import path

BUCKET_NAME = "age_detection_faehnrich"
STORAGE_LOCATION = 'raw_data'


def download():
    client = storage.Client().bucket(BUCKET_NAME)
    blob = client.blob(STORAGE_LOCATION + "/UTKFace.zip")
    print("Downloading...")
    blob.download_to_filename("../raw_data/Faces.zip")
    print("Downloading Done!")


def unzip():
    print("Unzipping...")
    with ZipFile('../raw_data/Faces.zip', 'r') as zip_ref:
        zip_ref.extractall("../raw_data/")
    print('Unzipping Done!')


def get_data():
    if not path.exists('../raw_data/Faces'):
        download()
        unzip()


if __name__ == '__main__':
    get_data()
