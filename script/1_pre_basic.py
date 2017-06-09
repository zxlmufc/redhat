# -*- coding: utf-8 -*-

import os
import re

import joblib
import pandas as pd

from script import utils

files = ['act_train.csv.zip', 'act_test.csv.zip', 'people.csv.zip']

all_data = []

for file in files:

    print('Loading %s ...' % file)
    data = utils.extract_zip_file(os.path.join(utils.root_path, 'data', file))
    data['date'] = data['date'].astype('datetime64')

    print('     shape: ', data.shape)
    print('')

    if file.startswith('act'):
        data.rename(columns=lambda x: re.sub('char', 'act_char', x), inplace=True)
        data.rename(columns=lambda x: re.sub('date', 'act_date', x), inplace=True)

        if file == 'act_test.csv.zip':
            data['outcome'] = -1

        all_data.append(data)

    else:
        data.rename(columns=lambda x: re.sub('char', 'ppl_char', x), inplace=True)
        data.rename(columns=lambda x: re.sub('date', 'reg_date', x), inplace=True)
        all_data = pd.concat(all_data, axis=0)

        print('Train & Test shape: ', all_data.shape)
        print('')

        all_data = pd.merge(all_data, data, on='people_id')

        print('All data shape: ', all_data.shape)


joblib.dump(all_data, os.path.join(utils.root_path, 'cache', 'all_data0'))
print("Done.")

