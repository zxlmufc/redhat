import os
import zipfile

import joblib
import pandas as pd

root_path = "/mnt/trident/xiaolan/python/kaggle/redhat"


def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    df = pd.read_csv(zf.open(zf.filelist[0].filename))
    return df


def load_dataset(preset, mode="eval"):
    df = joblib.load(os.path.join("../processed", preset['dataset']))
    features = preset['features']
    if mode =="eval":
        train_x = df.ix[df.tag == "train", features]
        train_y = df.ix[df.tag == "train", ['outcome']]
        test_x = df.ix[df.tag == 'test', features]
    return train_x, train_y, test_x

