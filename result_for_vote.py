import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

# test_correct_res = pd.read_csv('merge_data/result-iter=100-9116.csv')
# test_r1 = pd.read_csv('merge_data/result-time-one_hot-9068.csv')
# test_r2 = pd.read_csv('merge_data/result-wif_opt-9066.csv')
# test_r3 = pd.read_csv('merge_data/result-wifi_connect-9065.csv')
# test_r4 = pd.read_csv('merge_data/result_time_for_dinner-0_1-9064.csv')
test_correct_res = pd.read_csv('merge_data/result-test_for_train-9200.csv')
test_r1 = pd.read_csv('merge_data/result_11_11-2-9157.csv')
test_r2 = pd.read_csv('merge_data/result-merge-9192.csv')
test_r3 = pd.read_csv('merge_data/result_11_11-4-9157.csv')
test_r4 = pd.read_csv('merge_data/result-iter=100-9116.csv')
test_r5 = pd.read_csv('merge_data/result-time-one_hot-9068.csv')
test_r6 = pd.read_csv('merge_data/result-7_test_for_train-9167.csv')

test_correct_res.rename(columns={'shop_id': 'shop1_id'}, inplace=True)
test_r1.rename(columns={'shop_id': 'shop2_id'}, inplace=True)
test_r2.rename(columns={'shop_id': 'shop3_id'}, inplace=True)
test_r3.rename(columns={'shop_id': 'shop4_id'}, inplace=True)
test_r4.rename(columns={'shop_id': 'shop5_id'}, inplace=True)
test_r5.rename(columns={'shop_id': 'shop6_id'}, inplace=True)
test_r6.rename(columns={'shop_id': 'shop7_id'}, inplace=True)

test_correct_res = pd.merge(test_correct_res, test_r1[['row_id', 'shop2_id']], how='left', on='row_id')
test_correct_res = pd.merge(test_correct_res, test_r2[['row_id', 'shop3_id']], how='left', on='row_id')
test_correct_res = pd.merge(test_correct_res, test_r3[['row_id', 'shop4_id']], how='left', on='row_id')
test_correct_res = pd.merge(test_correct_res, test_r4[['row_id', 'shop5_id']], how='left', on='row_id')
test_correct_res = pd.merge(test_correct_res, test_r5[['row_id', 'shop6_id']], how='left', on='row_id')
test_correct_res = pd.merge(test_correct_res, test_r6[['row_id', 'shop7_id']], how='left', on='row_id')


def vote_for_result(x1, x2, x3, x4, x5, x6, x7):
    arr = [x1, x2, x3, x4, x5, x6, x7]
    # count the arr[i] happens in the arr by dict method
    res = {k: arr.count(k) for k in set(arr)}
    count = 0
    shop_id = None
    for (k, v) in res.items():
        if count < v:
            shop_id = k
            count = v
    if count >= 4:
        return shop_id
    else:
        return x1


test_correct_res['shop_id'] = list(
    map(lambda x1, x2, x3, x4, x5, x6, x7: vote_for_result(x1, x2, x3, x4, x5, x6, x7), test_correct_res['shop1_id'],
        test_correct_res['shop2_id'], test_correct_res['shop3_id'], test_correct_res['shop4_id'],
        test_correct_res['shop5_id'], test_correct_res['shop6_id'], test_correct_res['shop7_id']))
result = test_correct_res[['row_id', 'shop_id']]
# result = result[result['shop_id'] != 0]
result.to_csv("merge_data/result.csv", index=False)
