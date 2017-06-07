import zipfile
import pandas as pd
import os


root_path = "/mnt/trident/xiaolan/python/kaggle/redhat"

def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    fp = zf.extract(zf.filelist[0].filename)
    df = pd.read_csv(fp)
    return df

