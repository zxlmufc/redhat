import sys
import zipfile

import pandas as pd

root_path = "../"


def extract_zip_file(zip_path):
    zf = zipfile.ZipFile(zip_path)
    df = pd.read_csv(zf.open(zf.filelist[0].filename))
    return df

