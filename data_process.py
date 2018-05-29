# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

df = pd.read_csv('MetaData/on_train-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv('MetaData/off_train-ccf_first_round_shop_info.csv')
test = pd.read_csv('MetaData/AB-test-evaluation_public.csv')
test_result = pd.read_csv('merge_data/result.csv')
test_result_for_train = pd.merge(test, test_result[['row_id', 'shop_id']], how='left', on='row_id')
test_result_for_train.drop(['row_id', 'mall_id'], axis=1, inplace=True)
df = pd.concat((df, test_result_for_train), axis=0)
df = pd.merge(df, shop[['shop_id', 'mall_id']], how='left', on='shop_id')


# df['time_stamp_copy'] = pd.to_datetime(df['time_stamp'])

# df['time_stamp'] = df['time_stamp'].apply(parse)
# df['date'] = df['time_stamp_copy'].apply(datetime.date)
# df['time'] = df['time_stamp_copy'].apply(datetime.time)


# df['time'] = df.time.str.split(":")[0]
# df['time'] = df.time_stamp.apply(lambda s: int(s.split(' ')[1].split(':')[0]))
# df['week_day'] = df.time_stamp_copy.apply(lambda x: date.weekday(x) + 1)


def wifi_if_connect(s):
    s_arry = s.split(";")
    res = 0
    for i in range(len(s_arry)):
        temp = s_arry[i].split("|")
        cnect = temp[2]
        if cnect == "true":
            res = 1
            # res = int(temp[0].split("_")[1])
    return res


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


def get_wifi_info_optimized(s):
    temp_s = s.split(';')

    temp_strength = []
    wifi_infos = []

    for i in range(len(temp_s)):
        w = temp_s[i].split('|')
        temp_strength.append(int(w[1]))
        wifi_infos.append(Wifi_info(w[0], int(w[1]), w[2]))
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

    return best_id, second_id, ave, max_cha, square_error, third_id, fourth_id, fifth_id, sixth_id, seventh_id


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


# TIME FEATURE!
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


def is_weekend(s):
    day = int(s)
    if day > 5:
        return 1
    return 0


def chicken_dinner(s):
    time = int(s.split(" ")[1].split(":")[0])
    if (time >= 12) and (time <= 15):
        return 1
    elif (time >= 18) and (time <= 21):
        return 2
    else:
        return 0


train = pd.concat([df, test])

train['time'] = train['time_stamp'].apply(chicken_dinner)
train = pd.concat([train, pd.get_dummies(train['time'])], axis=1)

train['time_stamp'] = train['time_stamp'].apply(parse)
train['date'] = train['time_stamp'].apply(datetime.date)
train['is_weekend'] = train['date'].apply(lambda x: date.weekday(x) + 1)
train['is_weekend'] = train['is_weekend'].apply(lambda x: 1 if x in (6, 7) else 0)

train['wifi_connect'] = train['wifi_infos'].apply(wifi_if_connect)

# train = pd.concat([train, pd.get_dummies(train['wifi_connect'])], axis=1)
# ohe = preprocessing.OneHotEncoder()
# ohe.fit(list(train[['wifi_connect']].values))
# train['wifi_connect'] = ohe.transform(list(train[['wifi_connect']].values))

# train['time_stamp'] = train['time_stamp'].apply(divide_data_into_four_part)


train['wifi_infos_row'] = train['wifi_infos'].apply(get_wifi_info_optimized)

train['best_wifi'] = train['wifi_infos_row'].apply(get_best_one)
train['second_wifi'] = train['wifi_infos_row'].apply(get_second_one)
train['wifi_strength_ave'] = train['wifi_infos_row'].apply(get_wifi_strength_ave)
train['wifi_strength_max_cha'] = train['wifi_infos_row'].apply(get_wifi_strength_max_cha)
train['wifi_strength_square_error'] = train['wifi_infos_row'].apply(get_wifi_strength_square_error)
train['third_id'] = train['wifi_infos_row'].apply(get_third_id)
train['fourth_id'] = train['wifi_infos_row'].apply(get_fourth_id)
train['fifth_id'] = train['wifi_infos_row'].apply(get_fifth_id)
train['sixth_id'] = train['wifi_infos_row'].apply(get_sixth_id)
train['seventh_id'] = train['wifi_infos_row'].apply(get_seventh_id)
train['user_id'] = train['user_id'].apply(lambda x: int(x.split("_")[1]))
train.drop(['date', 'wifi_infos_row'], axis=1, inplace=True)
train.to_csv('transit_data/train.csv')
