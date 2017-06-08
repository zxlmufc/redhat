import zipfile
import pandas as pd
import sys
import os

root_path = "/mnt/trident/xiaolan/python/kaggle/redhat"


def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    df = pd.read_csv(zf.open(zf.filelist[0].filename))
    return df

