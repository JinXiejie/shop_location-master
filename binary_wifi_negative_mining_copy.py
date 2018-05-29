import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from datetime import time
import math

path = 'data/'
df = pd.read_csv(path + u'ccf_first_round_user_shop_behavior_train.csv')
shops = pd.read_csv(path + u'ccf_first_round_shop_info_train.csv')
test = pd.read_csv(path + u'evaluation_public.csv')
df = pd.merge(df, shops[['shop_id', 'mall_id']], how='left', on='shop_id')
df['label'] = 1
train = pd.concat([df, test])
mall_list = pd.read_csv('data/malls.csv')
mall_list = list(mall_list['0'])
result = pd.DataFrame()
mall_num = 0
for mall in mall_list:
    print(mall_num)
    print(mall)
    train1 = train[train.mall_id == mall].reset_index(drop=True)
    shop = shops[shops.mall_id == mall].reset_index(drop=True)
    l = []
    wifi_dict = {}
    for index, row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(r)
    delate_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            delate_wifi.append(i)
    m = []
    for row in l:
        new = {}
        for n in row.keys():
            if n not in delate_wifi:
                new[n] = row[n]
        m.append(new)
    train1 = pd.concat([train1, pd.DataFrame(m)], axis=1)
    list1 = list(train1[train1.shop_id.notnull()].dropna(axis=1, how='all').columns)
    list2 = list(train1[train1.shop_id.isnull()].dropna(axis=1, how='all').columns)
    intersection = list(set(list1).intersection(set(list2)))
    intersection.extend(['shop_id', 'row_id'])
    train1 = train1[intersection]

    shop_location = train1[['longitude', 'latitude', 'shop_id']]
    shop_location = shop_location[shop_location.shop_id.notnull()].groupby('shop_id').agg('mean').reset_index()
    shop_location.rename(columns={'longitude': 'r_longitude', 'latitude': 'r_latitude'}, inplace=True)
    shop = shop.merge(shop_location, on='shop_id', how='left')
    from math import radians, cos, sin, asin, sqrt


    def haversine(lon1, lat1, lon2, lat2):
        """ 
        Calculate the great circle distance between two points  
        on the earth (specified in decimal degrees) 
        """

        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine公式  
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000


    shop_info_1 = shop[['mall_id', 'longitude', 'latitude']]  # 加mall经纬度
    shop_info_1 = shop_info_1.groupby(['mall_id']).agg('mean').reset_index()
    shop_info_1.rename(columns={'longitude': 'mall_longitude', 'latitude': 'mall_latitude'}, inplace=True)
    shop = shop.merge(shop_info_1, on='mall_id', how='left')
    shop['s_m_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), shop['longitude'], shop['latitude'],
            shop['mall_longitude'], shop['mall_latitude']))
    shop = pd.concat([shop, pd.get_dummies(shop['category_id'])], axis=1)  #

    train1['time_stamp'] = train1['time_stamp'].apply(parse)
    train1['date'] = train1['time_stamp'].apply(datetime.date)
    train1['time'] = train1['time_stamp'].apply(datetime.time)
    train1['day_of_week'] = train1.date.apply(lambda x: date.weekday(x) + 1)
    train1['is_weekend'] = train1.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)


    def get_hour(t):
        h, m, s = t.split(':')
        return h


    train1['hour'] = train1['time'].apply(str).apply(get_hour).astype('int')


    def get_frame(t):
        if t < 11:
            f = 'f_1'
        elif t < 14:
            f = 'f_2'
        elif t < 18:
            f = 'f_3'
        else:
            f = 'f_4'
        return f


    train1['time_frame'] = train1['hour'].apply(get_frame)
    train1 = pd.concat([train1, pd.get_dummies(train1['time_frame'])], axis=1)
    train1.drop(['time'], axis=1, inplace=True)


    def get_day(t):
        y, m, d = t.split('-')
        return d


    train1['day'] = train1['date'].apply(str).apply(get_day).astype('int')
    train1.drop(['date'], axis=1, inplace=True)
    train1.drop(['time_stamp'], axis=1, inplace=True)
    # train_data
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()].reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_train = df_train.merge(shop, on='mall_id', how='left')
    df_train['u_s_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_train['longitude_x'], df_train['latitude_x'],
            df_train['longitude_y'], df_train['latitude_y']))
    df_train['u_r_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_train['longitude_x'], df_train['latitude_x'],
            df_train['r_longitude'], df_train['r_latitude']))
    df_train['label'] = list(map(lambda x, y: 1 if (x == y) else 0, df_train['shop_id_x'], df_train['shop_id_y']))
    df_train.drop(['shop_id_x'], axis=1, inplace=True)
    df_train.rename(columns={'shop_id_y': 'shop_id'}, inplace=True)
    df_train = pd.concat([df_train, pd.get_dummies(df_train['shop_id'])], axis=1)
    df_train_neg = df_train[df_train.label == 0]
    df_train_posi = df_train[df_train.label == 1]
    df_train = df_train_posi.append(df_train_neg.groupby('shop_id').apply(
        lambda t: t.sample(int(len(t) * 0.04), axis=0, random_state=1))).reset_index(drop=True)

    # 1st train 
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'max_depth': 7,
        'eval_metric': 'logloss',
        'seed': 0,
        'missing': -999,
        'silent': 1
    }
    feature = [x for x in df_train.columns if
               x not in ['user_id', 'category_id', 'time_frame', 'hour', 'u_s_distance', 'day', 'idx', 'label',
                         'shop_id', 'row_id', 'time_stamp', 'mall_id', 'wifi_infos', 'shop_id', 'mall_latitude',
                         'mall_longitude']]
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    # xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(xgbtrain, 'train')]
    num_rounds = 500
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=50)


    def predict(row):
        xgbtest = xgb.DMatrix(row[feature])
        row['pre_label'] = model.predict(xgbtest)
        return row


    # xgbtest = xgb.DMatrix(df_train_neg[feature])
    # df_train_neg['pre_label']=model.predict(xgbtest)
    df_train_neg = df_train_neg.groupby('shop_id').apply(predict)
    df_train_neg = df_train_neg[df_train_neg.pre_label > 0.3]
    df_train_neg_1 = df_train_neg[df_train_neg.pre_label > 0.5]
    df_train_neg_2 = df_train_neg[df_train_neg.pre_label > 0.95]
    df_train_posi = df_train_posi.groupby('shop_id').apply(predict)
    df_train_posi = df_train_posi[df_train_posi.pre_label < 0.9]
    # 2nd train 
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'auc',
        'seed': 0,  ###
        'missing': -999,
        'silent': 1,
        'subsample': 0.7,  #
        'scale_pos_weight': 4
    }
    df_train_again = pd.concat([df_train, df_train_neg, df_train_posi, df_train_neg_1, df_train_neg_2], axis=0).sample(
        frac=1).reset_index(drop=True)
    xgbtrain = xgb.DMatrix(df_train_again[feature], df_train_again['label'])
    # xgbval = xgb.DMatrix(df_train_val[feature],df_train_val['label'])
    watchlist = [(xgbtrain, 'train')]
    num_rounds = 400
    model_again = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=50)

    # validation
    df_test = df_test.merge(shop, on='mall_id', how='left')
    df_test['u_s_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_test['longitude_x'], df_test['latitude_x'],
            df_test['longitude_y'], df_test['latitude_y']))
    df_test['u_r_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_test['longitude_x'], df_test['latitude_x'],
            df_test['r_longitude'], df_test['r_latitude']))
    df_test.drop(['shop_id_x'], axis=1, inplace=True)
    df_test.rename(columns={'shop_id_y': 'shop_id'}, inplace=True)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['shop_id'])], axis=1)


    def predict_again(row):
        xgbtest = xgb.DMatrix(row[feature])
        row['label'] = model_again.predict(xgbtest)
        return row


    df_test = df_test.groupby('shop_id').apply(predict_again)
    r = df_test[['row_id', 'shop_id', 'label']]
    r = r.groupby('row_id').apply(lambda t: t[t.label == t.label.max()]).reset_index(drop=True)
    r.drop(['label'], axis=1, inplace=True)
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv(path + 'result_negative_mining.csv', index=False)
    mall_num += 1
