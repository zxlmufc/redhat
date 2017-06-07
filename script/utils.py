import zipfile
import pandas as pd
import os


def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    fp = zf.extract(zf.filelist[0].filename)
    df = pd.read_csv(fp)
    return df

