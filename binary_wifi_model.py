import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb

df = pd.read_csv('MetaData/on_train-ccf_first_round_user_shop_behavior.csv')
shops = pd.read_csv('MetaData/off_train-ccf_first_round_shop_info.csv')
test = pd.read_csv('MetaData/AB-test-evaluation_public.csv')
df = pd.merge(df, shops[['shop_id', 'mall_id']], how='left', on='shop_id')
df['label'] = 1
train = pd.concat([df, test])
mall_list = list(set(list(shops.mall_id)))
result = pd.DataFrame()
for mall in mall_list:
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
    from math import radians, cos, sin, asin, sqrt


    def haversine(lon1, lat1, lon2, lat2):
        """ 
        Calculate the great circle distance between two points  
        on the earth (specified in decimal degrees) 
        """

        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine format
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000


    shop_info_1 = shop[['mall_id', 'longitude', 'latitude']]  # add the logitude and latitude of the mall
    shop_info_1 = shop_info_1.groupby(['mall_id']).agg('mean').reset_index()
    shop_info_1.rename(columns={'longitude': 'mall_longitude', 'latitude': 'mall_latitude'}, inplace=True)
    shop = shop.merge(shop_info_1, on='mall_id', how='left')
    shop['s_m_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), shop['longitude'], shop['latitude'],
            shop['mall_longitude'], shop['mall_latitude']))
    shop['category_id'] = shop.category_id.apply(lambda x: x.split('_')[1]).astype('int')

    from datetime import datetime
    from dateutil.parser import parse
    from datetime import date
    from datetime import time

    train1['time_stamp'] = train1['time_stamp'].apply(parse)
    train1['date'] = train1['time_stamp'].apply(datetime.date)
    train1['time'] = train1['time_stamp'].apply(datetime.time)
    train1['day_of_week'] = train1.date.apply(lambda x: date.weekday(x) + 1)
    train1['is_weekend'] = train1.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)


    def get_hour(t):
        h, m, s = t.split(':')
        return h


    train1['hour'] = train1['time'].apply(str).apply(get_hour).astype('int')
    train1.drop(['time'], axis=1, inplace=True)


    def get_day(t):
        y, m, d = t.split('-')
        return d


    train1['day'] = train1['date'].apply(str).apply(get_day).astype('int')
    train1.drop(['date'], axis=1, inplace=True)
    train1.drop(['time_stamp'], axis=1, inplace=True)

    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()].reset_index(drop=True)

    df_train = df_train.reset_index()
    df_train.rename(columns={'index': 'idx'}, inplace=True)
    temp = df_train[['idx', 'mall_id']]
    temp = temp.merge(shop[['shop_id', 'mall_id']], on='mall_id', how='left')
    temp = temp.sample(n=20000, axis=0)
    temp.drop(['mall_id'], axis=1, inplace=True)
    temp = df_train.merge(temp, on='idx', how='right')
    temp['label'] = list(map(lambda x, y: 1 if (x == y) else 0, temp['shop_id_x'], temp['shop_id_y']))
    temp = temp[temp.label == 0]
    temp = temp.reset_index(drop=True)
    temp.drop(['shop_id_x'], axis=1, inplace=True)
    temp.rename(columns={'shop_id_y': 'shop_id'}, inplace=True)
    df_train = df_train.append(temp).reset_index(drop=True)
    df_train = df_train.merge(shop, on='shop_id', how='left')
    df_train.drop(['mall_id_x'], axis=1, inplace=True)
    df_train.rename(columns={'mall_id_y': 'mall_id'}, inplace=True)
    df_train['u_s_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_train['longitude_x'], df_train['latitude_x'],
            df_train['longitude_y'], df_train['latitude_y']))
    df_train = pd.concat([df_train, pd.get_dummies(df_train['shop_id'])], axis=1)
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'auc',
        'seed': 0,
        'missing': -999,
        'silent': 1
    }
    feature = [x for x in df_train.columns if
               x not in ['user_id', 'idx', 'label', 'shop_id', 'row_id', 'time_stamp', 'mall_id', 'wifi_infos',
                         'shop_id', 'mall_latitude', 'mall_longitude']]
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    # xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = 1000
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=50)

    df_test = df_test.merge(shop, on='mall_id', how='left')
    df_test['u_s_distance'] = list(
        map(lambda x1, y1, x2, y2: haversine(x1, y1, x2, y2), df_test['longitude_x'], df_test['latitude_x'],
            df_test['longitude_y'], df_test['latitude_y']))
    df_test.drop(['shop_id_x'], axis=1, inplace=True)
    df_test.rename(columns={'shop_id_y': 'shop_id'}, inplace=True)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['shop_id'])], axis=1)
    xgbtest = xgb.DMatrix(df_test[feature])
    df_test['label'] = model.predict(xgbtest)
    r = df_test[['row_id', 'shop_id', 'label']]
    r = r.groupby('row_id').apply(lambda t: t[t.label == t.label.max()]).reset_index(drop=True)
    r.drop(['label'], axis=1, inplace=True)
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')

result.to_csv("result.csv", index=False)
