# -*- coding: utf-8 -*-

import os

import joblib

from script import utils

data = joblib.load(os.path.join(utils.root_path, 'cache', 'all_data0'))


act_id = ['activity_id']
date = ['act_date', 'reg_date']
outcome = ['outcome']

cate_names = [name for name in data.columns if name not in act_id + date + outcome]


print('Encode feature ...')
for cate in cate_names:
    data[cate] = data[cate].astype('category').values.codes

data0 = data.drop(act_id + date + outcome, axis=1)


joblib.dump(data[act_id], os.path.join(utils.root_path, 'cache', 'activity_id'))
joblib.dump(data[date], os.path.join(utils.root_path, 'cache', 'date_column'))
joblib.dump(data[outcome], os.path.join(utils.root_path, 'cache', 'outcome'))
joblib.dump(data0, os.path.join(utils.root_path, 'cache', 'encode_feature'))

print("Done.")
