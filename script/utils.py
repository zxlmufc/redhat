import os
import zipfile

import joblib
import pandas as pd

root_path = "../"


def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    df = pd.read_csv(zf.open(zf.filelist[0].filename))
    return df


def extract_feature_names(preset):
    df = pd.DataFrame()

    for part in preset.get('features', []):
        df = pd.concat([df, joblib.load(os.path.join("../cache", part))], axis=1)

    # lp = 1
    # for pred in preset.get('predictions', []):
    #     if type(pred) is list:
    #         x.append('pred_%d' % lp)
    #         lp += 1
    #     else:
    #         x.append(pred)

    return df


def load_dataset(preset):

    x = extract_feature_names(preset)
    y = joblib.load(os.path.join("../cache", "outcome"))

    train_mask = y.values != -1

    train_x = x[train_mask]
    train_y = y[train_mask]
    test_x = x[~train_mask]

    print("Loading data: \n "
          "Dim train_x is %s; dim train_y is %s; dim test_x is %s" % (train_x.shape, train_y.shape, test_x.shape))

    return train_x, train_y, test_x
