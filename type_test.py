import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from enum import Enum

# shop_info_raw = pd.read_csv('MetaData/off_train-ccf_first_round_shop_info.csv')
user_info_raw = pd.read_csv('MetaData/on_train-ccf_first_round_user_shop_behavior.csv')
# evaluation_dataset = pd.read_csv('MetaData/AB-test-evaluation_public.csv')

# print shop_info_raw.shape
print user_info_raw.shape
# print evaluation_dataset.shape


def get_best_wifi(s):
    s = s.split(';')[0]
    # print s
    s = int(s.split('|')[1])
    # print type(s)
    return s

def wifi(s):
    # print type(s)
    return s

wifi_infos = user_info_raw.wifi_infos.apply(get_best_wifi)
wifi_infos.astype("int32")
print wifi_infos.dtype
wifi_infos = user_info_raw.wifi_infos.apply(wifi)
# print type(wifi_infos)


