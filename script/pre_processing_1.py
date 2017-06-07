from script import utils
import pandas as pd
import joblib

act_test_data = utils.extract_zip_file('/mnt/trident/xiaolan/python/kaggle/redhat/data/act_test.csv.zip')
act_train_data = utils.extract_zip_file('/mnt/trident/xiaolan/python/kaggle/redhat/data/act_train.csv.zip')
people_data = utils.extract_zip_file('/mnt/trident/xiaolan/python/kaggle/redhat/data/people.csv.zip')
act_train_data['tag'] = 'train'
act_test_data['tag'] = 'test'
act_test_data['outcome'] = -1

act_all_data = pd.concat([act_train_data, act_test_data], axis=0)
act_all_data['date'] = act_all_data['date'].astype('datetime64')

