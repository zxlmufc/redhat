from script import utils
import pandas as pd
import os
import re
import joblib

act_test_data = utils.extract_zip_file(os.path.join(utils.root_path, 'data/act_test.csv.zip'))
act_train_data = utils.extract_zip_file(os.path.join(utils.root_path, 'data/act_train.csv.zip'))
people_data = utils.extract_zip_file(os.path.join(utils.root_path, 'data/people.csv.zip'))
act_train_data['tag'] = 'train'
act_test_data['tag'] = 'test'
act_test_data['outcome'] = -1
act_all_data = pd.concat([act_train_data, act_test_data], axis=0)
act_all_data['date'] = act_all_data['date'].astype('datetime64')

act_all_data = act_all_data.rename(columns=lambda x: re.sub('char', 'act_char', x))
people_data = people_data.rename(columns=lambda x: re.sub('char', 'people_char', x))

all_data0 = pd.merge(act_all_data, people_data, left_on='people_id', right_on='people_id')

joblib.dump(all_data0, os.path.join(utils.root_path, 'processed/all_data0'))
