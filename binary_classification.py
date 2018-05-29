import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from enum import Enum

shop_info_raw = pd.read_csv('MetaData/off_train-ccf_first_round_shop_info.csv')
user_info_raw = pd.read_csv('MetaData/on_train-ccf_first_round_user_shop_behavior.csv')
evaluation_dataset = pd.read_csv('MetaData/AB-test-evaluation_public.csv')

print shop_info_raw.shape
print user_info_raw.shape
print evaluation_dataset.shape

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


def divide_AugTime_into_weeknums(s):
    day = int(s.split(' ')[0].split('-')[2])
    if day == 7 | day == 14 | day == 21 | day == 28:
        return 1
    if day == 1 | day == 8 | day == 15 | day == 22 | day == 29:
        return 2
    if day == 2 | day == 9 | day == 16 | day == 23 | day == 30:
        return 3
    if day == 3 | day == 10 | day == 17 | day == 24 | day == 31:
        return 4
    if day == 4 | day == 11 | day == 18 | day == 25:
        return 5
    if day == 5 | day == 12 | day == 19 | day == 26:
        return 6
    if day == 6 | day == 13 | day == 20 | day == 27:
        return 7


def divide_SepTime_into_weeknums(s):
    day = int(s.split(' ')[0].split('-')[2])
    if day == 7 | day == 14 | day == 21 | day == 28:
        return 1
    if day == 1 | day == 8 | day == 15 | day == 22 | day == 29:
        return 2
    if day == 2 | day == 9 | day == 16 | day == 23 | day == 30:
        return 3
    if day == 3 | day == 10 | day == 17 | day == 24 | day == 31:
        return 2
    if day == 4 | day == 11 | day == 18 | day == 25:
        return 1
    if day == 5 | day == 12 | day == 19 | day == 26:
        return 6
    if day == 6 | day == 13 | day == 20 | day == 27:
        return 7


user_info_labeled['time_stamp'] = user_info_labeled.time_stamp.apply(divide_data_into_four_part)

# divide the data into two part
user_info_eval = user_info_labeled[user_info_labeled['time_stamp'] == 0]
user_info_train = user_info_labeled[user_info_labeled['time_stamp'] > 0]


# judge if users connect the wifi(True or False),transform it by half of strength adding to,
# then we will get the best wifi in single sample
def get_real_best_wifi(s):
    s = s.split(';')
    real_best_wifi = ''
    strength = -999
    for i in range(len(s)):
        w = s[i].split('|')
        if w[2] == 'false':
            if int(w[1]) > strength:
                real_best_wifi = w[0]
                strength = int(w[1])
        if w[2] == 'true':
            w_true = int(w[1]) + (int(w[1])) / 2
            if w_true > strength:
                real_best_wifi = w[0]
                strength = w_true
    real_best_wifi = real_best_wifi.split('_')
    return int(real_best_wifi[1])


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


def get_best_second_optimized(s):
    temp_s = s.split(';')

    temp_strength = []

    for i in range(len(temp_s)):
        w = temp_s[i].split('|')
        temp_strength.append(int(w[1]))
        print(i)

    temp_strength.sort(reverse=True)

    temp_best = []
    temp_second = []
    best_s = temp_strength[0]
    for i in range(len(temp_strength)):
        if temp_strength[i] == best_s:
            temp_best.append(temp_strength[i])
            temp_strength.remove(temp_strength[i])
        else:
            break

    second_s = temp_strength[0]
    for i in range(len(temp_strength)):
        if temp_strength[i] == second_s:
            temp_second.append(temp_strength[i])
            temp_strength.remove(temp_strength[i])
        else:
            break

    temp_best_id = []
    temp_second_id = []
    for i in range(len(temp_s)):
        for j in range(len(temp_best)):
            if temp_s[i].find(str(temp_best[j])) > -1:
                temp_best_id.append(int(temp_s[i].split('|')[0].split('_')[1]))

        for j in range(len(temp_second)):
            if temp_s[i].find(str(temp_second[j])) > -1:
                temp_second_id.append(int(temp_s[i].split('|')[0].split('_')[1]))

    temp_best_id.sort(reverse=True)
    temp_second_id.sort(reverse=True)

    return temp_best_id[0], temp_second_id[0]


# USER FEATURE!


# WIFI FEATURE!
# user_info_labeled['wifi_infos'] = user_info_labeled.wifi_infos.apply(get_best_wifi)

train_wifi_infos = user_info_labeled[['wifi_infos']]
# user_info_labeled['wifi_infos'] = train_wifi_infos.wifi_infos.apply(get_best_wifi)
# user_info_labeled['wifi_infos_1'] = train_wifi_infos.wifi_infos.apply(get_real_best_wifi)


shop_mall = shop_info_raw[['shop_id', 'mall_id']]
shop_mall['shop_id'] = mainLabelEncoder.fit_transform(shop_info_raw['shop_id'])
shop_mall.rename(columns={'shop_id': 'shop_label'}, inplace=True)

user_info_labeled = pd.merge(user_info_labeled, shop_mall, on=['shop_label'], how='left')

params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
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


# transform the db number [-99, -5] to signal amplitude [1,56230]
def db_to_amp(s):
    db_num = float(s.split('|')[1])
    db_num = db_num / 20.0
    amp_num = 100000.0 * (10.0 ** db_num)
    return int(amp_num)


def float_to_int(f):
    return int(f)


models = []
labelEncoders = []
result = pd.DataFrame()
# train_out = pd.DataFrame()

for mall in malls['mall_id']:
    print(mall)
    # prepare data in this mall
    dataset_for_train = user_info_labeled[user_info_labeled['mall_id'] == mall]
    dataset_for_train_x = dataset_for_train[['longitude', 'latitude', 'wifi_infos']]
    for model_id in dataset_for_train['shop_label']:
        each_train = dataset_for_train[['shop_label']]
        dataset_for_train_y = each_train
    dataset_for_train_y = dataset_for_train[['shop_label']]
    # init shop labelEncoder for this mall
    temp_le = LabelEncoder()
    dataset_for_train_y = temp_le.fit_transform(dataset_for_train_y['shop_label'])
    # save the labelEncoder so as to decode
    labelEncoders.append(temp_le)
    # get and set num_class
    params['num_class'] = len(shop_mall[shop_mall['mall_id'] == mall])

    xgb_data = xgb.DMatrix(dataset_for_train_x, label=dataset_for_train_y)
    watchlist = [(xgb_data, 'train')]
    model = xgb.train(params, xgb_data, num_boost_round=50, evals=watchlist)
    models.append(model)

# the mall_ids are given
# handle the predict data

predict_malls = evaluation_dataset[['mall_id']]
predict_malls.drop_duplicates(inplace=True)


def float_to_int(f):
    return int(f)


result = pd.DataFrame()

for local_mall in predict_malls['mall_id']:
    print(local_mall)
    predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == local_mall]
    predict_data_id = predict_data[['row_id']]
    # predict_data_mall = predict_data[['mall_id']]
    predict_data.drop(['user_id', 'time_stamp', 'row_id', 'mall_id'], axis=1, inplace=True)
    predict_data['wifi_infos'] = predict_data.wifi_infos.apply(get_best_wifi)

    xgb_predict_data = xgb.DMatrix(predict_data)
    # get the index of this mall so as to get its model and labelEncoder
    index_of_mall = malls[malls['mall_id'] == local_mall].index.tolist()[0]
    predict_data['label'] = models[index_of_mall].predict(xgb_predict_data)
    predict_data['label'] = predict_data.label.apply(float_to_int)
    predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
    # turn to the main encoder
    predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
    predict_data_id['shop_id'] = predict_data['label']
    result = pd.concat([result, predict_data_id], axis=0)

result.to_csv("result.csv", index=None)
