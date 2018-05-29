#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:42:24 2017

@author: lab-tan.yun
"""

import pandas as pd
# import lightgbm as lgb
import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from enum import Enum

shop_info_raw = pd.read_csv('data/ccf_first_round_shop_info.csv')
user_info_raw = pd.read_csv('data/ccf_first_round_user_shop_behavior.csv')
evaluation_dataset = pd.read_csv('data/evaluation_public.csv')
# get mall_id
malls = shop_info_raw[['mall_id']]
malls.drop_duplicates(inplace=True)
malls.reset_index(inplace=True)
malls = malls.drop(['index'], axis=1)

# get shop label
mainLabelEncoder = LabelEncoder()
shop_label = shop_info_raw[['shop_id']]
mainLabelEncoder.fit(shop_label)

user_in_shop = user_info_raw[['shop_id']]
user_in_shop = mainLabelEncoder.transform(user_in_shop)

user_info_labeled = user_info_raw
user_info_labeled['shop_label'] = user_in_shop

# user_info_labeled = user_info_labeled.drop(['shop_id','user_id','time_stamp'],axis = 1)
user_info_labeled = user_info_labeled.drop(['shop_id', 'user_id'], axis=1)


def divide_data_into_four_part(s):
    day = int(s.split(' ')[0].split('-')[2]) % 4
    if day == 0:
        return 0
    elif day == 1:
        return 1
    elif day == 2:
        return 2
    elif day == 3:
        return 3


user_info_labeled['time_stamp'] = user_info_labeled.time_stamp.apply(divide_data_into_four_part)


# divide the data into two part
# user_info_eval = user_info_labeled[user_info_labeled['time_stamp'] == 0]
# user_info_train = user_info_labeled[user_info_labeled['time_stamp'] > 0]


# as i think, if we only use the best one, it may be over fitting
def get_best_wifi(s):
    s = s.split(';')
    best_wifi = ''
    strength = -999
    for i in range(len(s)):
        w = s[i].split('|')
        if int(w[1]) > strength:
            best_wifi = w[0]
            strength = int(w[1])
    best_wifi = best_wifi.split('_')
    return int(best_wifi[1])


# create a wifi_info class to manage the wifi info
class Wifi_info(object):
    bssid = ''
    strength = -999
    connected = 0

    def __init__(self, b, s, c):
        self.bssid = int(b.split('_')[1])
        self.strength = s
        if c == 'true':
            self.connected = 1
        else:
            self.connected = 0

    def get_strength(self):
        return self.strength

    def get_bssid(self):
        return self.bssid


# create a wifi info num
class wifi_info_num(Enum):
    best_id = 0
    second_id = 1
    ave = 2
    max_cha = 3
    square_error = 4
    third_id = 5
    fourth_id = 6
    fifth_id = 7
    sixth_id = 8
    seventh_id = 9
    connect_wifi = 10


# unfinished,if the wifi signal are too close or even they have the same strength.
# the trouble will come.
# im not sure if this will cause over fitting.
# Stop doing this , try the simplest one.
def get_wifi_info_optimized(s):
    temp_s = s.split(';')

    temp_strength = []
    wifi_infos = []

    connect_wifi = 0
    for i in range(len(temp_s)):
        w = temp_s[i].split('|')
        temp_strength.append(int(w[1]))
        wifi_infos.append(Wifi_info(w[0], int(w[1]), w[2]))
        if w[2] == 'true':
            connect_wifi = int(w[0].split('_')[1])

            # print(i)

    temp_strength.sort(reverse=True)
    wifi_infos.sort(key=Wifi_info.get_strength, reverse=True)

    ave = sum(temp_strength) / len(temp_strength)

    max_cha = temp_strength[len(temp_strength) - 1] - temp_strength[0]

    square_error = 0
    for i in range(len(temp_strength)):
        square_error = square_error + (temp_strength[i] - ave) * (temp_strength[i] - ave)

    square_error = square_error / len(temp_strength)

    #    best_s = temp_strength[0]

    #    best_s = wifi_infos[0].get_strength()
    #    try:
    #        second_s = temp_strength[1]
    #    except IndexError:
    #        second_s = -1

    # we suppose to get the closet group.
    # but wait, we should  do the simple one first.

    #    best_id = 0
    #    second_id = 0

    #    for i in range(len(temp_s)):
    #        if temp_s[i].find(str(best_s)) > -1:
    #            best_id = int(temp_s[i].split('|')[0].split('_')[1])
    #            break;
    #
    #    for i in range(len(temp_s)):
    #        if temp_s[i].find(str(second_s)) > -1:
    #            second_id = int(temp_s[i].split('|')[0].split('_')[1])

    # now the best and the second id all get

    #    best_wifi = best_wifi.split('_')
    #    if second_wifi != 'null':
    #        second_wifi = second_wifi.split('_')
    #        return int(best_wifi[1]),int(second_wifi[1])
    # keep relax....Think about the wonderful life.

    best_id = wifi_infos[0].get_bssid()
    try:
        second_id = wifi_infos[1].get_bssid()
    except IndexError:
        second_id = -1

    try:
        third_id = wifi_infos[2].get_bssid()
    except IndexError:
        third_id = -1

    try:
        fourth_id = wifi_infos[3].get_bssid()
    except IndexError:
        fourth_id = -1
    try:
        fifth_id = wifi_infos[4].get_bssid()
    except IndexError:
        fifth_id = -1
    try:
        sixth_id = wifi_infos[5].get_bssid()
    except IndexError:
        sixth_id = -1
    try:
        seventh_id = wifi_infos[6].get_bssid()
    except IndexError:
        seventh_id = -1

    return best_id, second_id, ave, max_cha, square_error, third_id, fourth_id, fifth_id, sixth_id, seventh_id, connect_wifi


user_info_labeled['wifi_infos_raw'] = user_info_labeled['wifi_infos']

user_info_labeled['wifi_infos'] = user_info_labeled.wifi_infos.apply(get_wifi_info_optimized)


def get_best_one(s):
    return s[0]


def get_second_one(s):
    return s[1]


def get_wifi_strength_ave(s):
    return s[2]


def get_wifi_strength_max_cha(s):
    return s[3]


def get_wifi_strength_square_error(s):
    return s[4]


def get_third_id(s):
    return s[wifi_info_num.third_id.value]


def get_fourth_id(s):
    return s[wifi_info_num.fourth_id.value]


def get_fifth_id(s):
    return s[wifi_info_num.fifth_id.value]


def get_sixth_id(s):
    return s[wifi_info_num.sixth_id.value]


def get_seventh_id(s):
    return s[wifi_info_num.seventh_id.value]


def get_connect_wifi(s):
    return s[wifi_info_num.connect_wifi.value]


user_info_labeled['best_wifi'] = user_info_labeled.wifi_infos.apply(get_best_one)
user_info_labeled['second_wifi'] = user_info_labeled.wifi_infos.apply(get_second_one)
user_info_labeled['wifi_strength_ave'] = user_info_labeled.wifi_infos.apply(get_wifi_strength_ave)
user_info_labeled['wifi_strength_max_cha'] = user_info_labeled.wifi_infos.apply(get_wifi_strength_max_cha)
# user_info_labeled['wifi_strength_square_error'] = user_info_labeled.wifi_infos.apply(get_wifi_strength_square_error)

# new add in 10.25
user_info_labeled['third_id'] = user_info_labeled.wifi_infos.apply(get_third_id)
user_info_labeled['fourth_id'] = user_info_labeled.wifi_infos.apply(get_fourth_id)
user_info_labeled['fifth_id'] = user_info_labeled.wifi_infos.apply(get_fifth_id)
user_info_labeled['sixth_id'] = user_info_labeled.wifi_infos.apply(get_sixth_id)
user_info_labeled['seventh_id'] = user_info_labeled.wifi_infos.apply(get_seventh_id)
user_info_labeled['connect_wifi'] = user_info_labeled.wifi_infos.apply(get_connect_wifi)

# user_info_labeled.drop(['wifi_infos'],axis = 1,inplace = True)

shop_mall = shop_info_raw[['shop_id', 'mall_id']]
shop_mall['shop_id'] = mainLabelEncoder.fit_transform(shop_info_raw['shop_id'])
shop_mall.rename(columns={'shop_id': 'shop_label'}, inplace=True)
user_info_labeled = pd.merge(user_info_labeled, shop_mall, on=['shop_label'], how='left')

# at here,we are going to train our models
# we need different encoder for different mall

# the common parameter for all models
# exclude num_class,becasue this parameter is unique in each mall
params = {'booster': 'gbtree',
          'objective': 'multi:softmax',
          'eval_metric': 'merror',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 6,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.1,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }

# lgb params
# params = {
#    'task': 'train',
#    'application': 'multiclass',
#    'boosting_type': 'gbdt',
#    'metric': 'multi_error',
#    'num_leaves': 31,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 0
#    
# }


models = []
labelEncoders = []
# malls.reset_index(inplace=True)

predict_malls = evaluation_dataset[['mall_id']]
predict_malls.drop_duplicates(inplace=True)


def float_to_int(f):
    return int(f)


result = pd.DataFrame()

for mall in malls['mall_id']:
    print(mall)
    print(malls[malls['mall_id'] == mall].index[0])
    # prepare data in this mall
    # init shop labelEncoder for this mall
    temp_le = LabelEncoder()
    dataset_in_this_mall = user_info_labeled[user_info_labeled['mall_id'] == mall]
    dataset_in_this_mall['shop_label'] = temp_le.fit_transform(dataset_in_this_mall['shop_label'])

    predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == mall]
    predict_data_id = predict_data[['row_id']]

    predict_data_id.reset_index(inplace=True)
    predict_data_id.drop(['index'], axis=1, inplace=True)

    # predict_data_mall = predict_data[['mall_id']]
    predict_data.drop(['user_id', 'row_id'], axis=1, inplace=True)

    predict_data['wifi_infos_raw'] = predict_data['wifi_infos']

    predict_data['wifi_infos'] = predict_data.wifi_infos.apply(get_wifi_info_optimized)

    predict_data['best_wifi'] = predict_data.wifi_infos.apply(get_best_one)
    predict_data['second_wifi'] = predict_data.wifi_infos.apply(get_second_one)
    predict_data['wifi_strength_ave'] = predict_data.wifi_infos.apply(get_wifi_strength_ave)
    predict_data['wifi_strength_max_cha'] = predict_data.wifi_infos.apply(get_wifi_strength_max_cha)
    #    predict_data['wifi_strength_square_error'] = predict_data.wifi_infos.apply(get_wifi_strength_square_error)

    # new add in 10.25
    predict_data['third_id'] = predict_data.wifi_infos.apply(get_third_id)
    predict_data['fourth_id'] = predict_data.wifi_infos.apply(get_fourth_id)
    predict_data['fifth_id'] = predict_data.wifi_infos.apply(get_fifth_id)
    predict_data['sixth_id'] = predict_data.wifi_infos.apply(get_sixth_id)
    predict_data['seventh_id'] = predict_data.wifi_infos.apply(get_seventh_id)
    predict_data['connect_wifi'] = predict_data.wifi_infos.apply(get_connect_wifi)

    predict_data['shop_label'] = 1

    predict_data.reset_index(inplace=True)
    predict_data.drop(['index'], axis=1, inplace=True)

    dataset_in_this_mall.reset_index(inplace=True)
    dataset_in_this_mall.drop(['index'], axis=1, inplace=True)

    #    predict_data.drop(['wifi_infos'],axis = 1,inplace = True)



    combined_data = pd.concat((dataset_in_this_mall, predict_data), axis=0)

    combined_data.reset_index(inplace=True)
    combined_data.drop(['index'], axis=1, inplace=True)

    # get sparse matrix
    data = combined_data['wifi_infos_raw'].str.split(';', expand=True).stack().reset_index()

    #    data = dataset_in_this_mall['wifi_infos_raw'].str.split(';',expand = True).stack().reset_index()

    data.rename(columns={'level_0': 'record_id', 'level_1': 'shit', 0: 'wifi'}, inplace=True)
    data['wifi_id'] = data['wifi'].apply(lambda x: x.split('|')[0])
    data['wifi_strength'] = data['wifi'].apply(lambda x: int(x.split('|')[1]))
    data.drop(['shit', 'wifi'], axis=1, inplace=True)
    data = data.groupby(['record_id', 'wifi_id'])['wifi_strength'].agg('mean').reset_index()
    data.set_index(['record_id', 'wifi_id'], inplace=True)
    sparse_matrix = data.unstack(level=-1)

    all_train_wifi_data_sparse = sparse_matrix[0:dataset_in_this_mall.shape[0]]

    predict_wifi_data_sparse = sparse_matrix[dataset_in_this_mall.shape[0]:sparse_matrix.shape[0]]

    sparse_matrix = None

    predict_wifi_data_sparse.reset_index(inplace=True)

    predict_wifi_data_sparse.drop(['record_id'], axis=1, inplace=True)

    test_df = pd.concat((dataset_in_this_mall, all_train_wifi_data_sparse), axis=1)

    dataset_in_this_mall = None
    all_train_wifi_data_sparse = None

    final_predict_data = pd.concat((predict_data, predict_wifi_data_sparse), axis=1)

    predict_wifi_data_sparse = None

    data_for_predict = test_df[test_df['time_stamp'] == 0]

    data_for_train = test_df[test_df['time_stamp'] > 0]

    test_df = None

    data_for_train_x = data_for_train.drop(['time_stamp', 'shop_label', 'wifi_infos', 'wifi_infos_raw', 'mall_id'],
                                           axis=1)
    data_for_train_y = data_for_train[['shop_label']].astype('int')

    data_for_predict_x = data_for_predict.drop(['time_stamp', 'shop_label', 'wifi_infos', 'wifi_infos_raw', 'mall_id'],
                                               axis=1)
    data_for_predict_y = data_for_predict[['shop_label']].astype('int')

    final_predict_data = final_predict_data.drop(
        ['time_stamp', 'shop_label', 'wifi_infos', 'wifi_infos_raw', 'mall_id'], axis=1)

    # save the labelEncoder so as to decode
    labelEncoders.append(temp_le)

    #    data_for_predict = dataset_in_this_mall[user_info_labeled['time_stamp'] == 0]
    #
    #    data_for_train = dataset_in_this_mall[user_info_labeled['time_stamp'] > 0]
    #
    #    data_for_train_x = data_for_train.drop(['time_stamp','shop_label','mall_id'],axis = 1)
    #    data_for_train_y = data_for_train[['shop_label']]
    #
    #    data_for_predict_x = data_for_predict.drop(['time_stamp','shop_label','mall_id'],axis = 1)
    #    data_for_predict_y = data_for_predict[['shop_label']]

    # get and set num_class
    params['num_class'] = len(shop_mall[shop_mall['mall_id'] == mall])

    data_for_train_x.fillna(0, inplace=True)
    data_for_predict_x.fillna(0, inplace=True)
    final_predict_data.fillna(0, inplace=True)

    csr = csr_matrix(data_for_train_x.values)

    csr_predict = csr_matrix(data_for_predict_x.values)

    csr_final_predict = csr_matrix(final_predict_data.values)
    #
    xgb_train_data = xgb.DMatrix(csr, label=data_for_train_y)

    xgb_predict_data = xgb.DMatrix(csr_predict, label=data_for_predict_y)

    xgb_final_predict_data = xgb.DMatrix(csr_final_predict)
    # create dataset for lightgbm

    #    lgb_train = lgb.Dataset(data_for_train_x.values, data_for_train_y.values.reshape(-1))
    #
    #    lgb_eval = lgb.Dataset(data_for_predict_x.values, data_for_predict_y.values.reshape(-1), reference=lgb_train)
    #
    watchlist = [(xgb_train_data, 'train'), (xgb_predict_data, 'val')]

    model = xgb.train(params, xgb_train_data, num_boost_round=500, early_stopping_rounds=20, evals=watchlist)

    #    model = lgb.train(params,lgb_train,num_boost_round = 500,early_stopping_rounds = 20,valid_sets=lgb_eval)





    index_of_mall = malls[malls['mall_id'] == mall].index.tolist()[0]

    predicted_label = model.predict(csr_final_predict, num_iteration=model.best_iteration)

    real_predict_label = []

    for i in range(len(predicted_label)):
        real_predict_label.append(np.where(predicted_label[i] == predicted_label[i].max())[0][0])

    label_dateframe = pd.DataFrame(real_predict_label)
    #    predict_data['label'] = model.predict(csr_final_predict)

    predict_data['label'] = label_dateframe
    predict_data['label'] = temp_le.inverse_transform(predict_data['label'])
    # turn to the main encoder
    predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
    predict_data_id['shop_id'] = predict_data['label']
    result = pd.concat([result, predict_data_id], axis=0)
    result.to_csv("result.csv", index=None)
    model.save_model('xgb_model/model' + mall + '.txt', num_iteration=model.best_iteration)
    models.append(model)

# the mall_ids are given
# handle the predict data

# predict_malls = evaluation_dataset[['mall_id']]
# predict_malls.drop_duplicates(inplace = True)
#
# def float_to_int(f):
#    return int(f)
#
#
#
# result = pd.DataFrame()
#
# for local_mall in predict_malls['mall_id']:
#    print(local_mall)
#    
#    predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == local_mall]
#    predict_data_id = predict_data[['row_id']]
#    #predict_data_mall = predict_data[['mall_id']]
#    predict_data.drop(['user_id','time_stamp','row_id','mall_id'],axis = 1,inplace=True)
#    
#    predict_data['wifi_infos_raw'] = predict_data['wifi_infos']
#    
#    predict_data['wifi_infos'] = predict_data.wifi_infos.apply(get_wifi_info_optimized)
#    
#    predict_data['best_wifi'] = predict_data.wifi_infos.apply(get_best_one)
#    predict_data['second_wifi'] = predict_data.wifi_infos.apply(get_second_one)
#    predict_data['wifi_strength_ave'] = predict_data.wifi_infos.apply(get_wifi_strength_ave)
#    predict_data['wifi_strength_max_cha'] = predict_data.wifi_infos.apply(get_wifi_strength_max_cha)
#    predict_data['wifi_strength_square_error'] = predict_data.wifi_infos.apply(get_wifi_strength_square_error)
#    
#    #new add in 10.25
#    predict_data['third_id'] = predict_data.wifi_infos.apply(get_third_id)
#    predict_data['fourth_id'] = predict_data.wifi_infos.apply(get_fourth_id)
#    predict_data['fifth_id'] = predict_data.wifi_infos.apply(get_fifth_id)
#    predict_data['sixth_id'] = predict_data.wifi_infos.apply(get_sixth_id)
#    predict_data['seventh_id'] = predict_data.wifi_infos.apply(get_seventh_id)
#
#    
#    predict_data.drop(['wifi_infos'],axis = 1,inplace = True)
#
#    data = predict_data['wifi_infos_raw'].str.split(';',expand = True).stack().reset_index()
#    
#    data.rename(columns = {'level_0':'record_id','level_1':'shit',0:'wifi'},inplace=True)
#    data['wifi_id']=data['wifi'].apply(lambda x: x.split('|')[0])
#    data['wifi_strength'] = data['wifi'].apply(lambda x: int(x.split('|')[1]))
#    data.drop(['shit','wifi'],axis = 1,inplace = True)
#    data = data.groupby(['record_id','wifi_id'])['wifi_strength'].agg('mean').reset_index()
#    data.set_index(['record_id','wifi_id'],inplace=True)
#    sparse_matrix = data.unstack(level = -1)
#    
#    dataset_in_this_mall.reset_index(inplace=True)
#    dataset_in_this_mall.drop(['index'],axis = 1,inplace = True)
#    
#    test_dataset_in_this_mall = pd.concat((predict_data,sparse_matrix),axis = 1)
#
#    test_df = test_dataset_in_this_mall
#    
#    test_df.drop(['wifi_infos_raw'],axis = 1,inplace = True)
#    
#    
#    test_df.fillna(0,inplace=True)
#    
#    predict_data = csr_matrix(test_df.values)
#    
#    
#    xgb_predict_data = xgb.DMatrix(predict_data)
#    #get the index of this mall so as to get its model and labelEncoder
#    index_of_mall = malls[malls['mall_id'] == local_mall].index.tolist()[0]
#    predict_data['label'] = models[index_of_mall].predict(xgb_predict_data)
#    predict_data['label'] = predict_data.label.apply(float_to_int)
#    predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
#    #turn to the main encoder
#    predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
#    predict_data_id['shop_id'] = predict_data['label']
#    result = pd.concat([result,predict_data_id],axis = 0)

result.to_csv("result.csv", index=None)
