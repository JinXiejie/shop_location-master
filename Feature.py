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
            w_true = int(w[1]) + (int(w[1]))/2
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

def float_to_int(f):
    return int(f)

# models = []
labelEncoders = []
result = pd.DataFrame()
# train_out = pd.DataFrame()

for mall in malls['mall_id']:
    print(mall)
    # prepare data in this mall
    # get the sparse matrix of dataset_for_train
    dataset_for_train = user_info_labeled[user_info_labeled['mall_id'] == mall]
    # dataset_for_train_x = dataset_for_train[['longitude', 'latitude', 'wifi_infos', 'wifi_infos_1']]
    # dataset_for_train_y = dataset_for_train[['shop_label']]
    # init shop labelEncoder for this mall
    temp_le = LabelEncoder()
    dataset_for_train['shop_label'] = temp_le.fit_transform(dataset_for_train['shop_label'])

    # build sparse matrix for wifi_infos
    wifi_infos_sparse_matrix_train = dataset_for_train['wifi_infos'].str.split(";", expand=True).stack().reset_index()
    wifi_infos_sparse_matrix_train.rename(columns={'level_0': 'record_id', 'level_1': 'drop_id', 0: 'new_wifi_infos'},
                                         inplace=True)
    wifi_infos_sparse_matrix_train['wifi_id'] = wifi_infos_sparse_matrix_train['new_wifi_infos'].apply(lambda x: x.split('|')[0])
    wifi_infos_sparse_matrix_train['signal'] = wifi_infos_sparse_matrix_train['new_wifi_infos'].apply(lambda x: int(x.split('|')[1]))
    wifi_infos_sparse_matrix_train.drop(['drop_id', 'new_wifi_infos'], axis=1, inplace=True)
    wifi_infos_sparse_matrix_train = wifi_infos_sparse_matrix_train.groupby(['record_id', 'wifi_id'])['signal'].agg('mean').reset_index()
    wifi_infos_sparse_matrix_train.set_index(['record_id', 'wifi_id'], inplace=True)
    sparse_matrix_train = wifi_infos_sparse_matrix_train.unstack(level = -1)
    len_of_train = len(sparse_matrix_train)



    train_out = sparse_matrix_train
    name_out = mall.str + ".csv"
    train_out.to_csv(name_out, index=None)



    # get the sparse matrix of predict_data
    predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == mall]
    predict_data_id = predict_data[['row_id']]
    # predict_data_mall = predict_data[['mall_id']]
    predict_data.drop(['user_id', 'time_stamp', 'row_id', 'mall_id'], axis=1, inplace=True)

    # predict_data['wifi_infos_raw'] = predict_data['wifi_infos']

    # predict_data.drop(['wifi_infos'], axis=1, inplace=True)

    wifi_infos_sparse_matrix_test = predict_data['wifi_infos'].str.split(";", expand=True).stack().reset_index()
    wifi_infos_sparse_matrix_test.rename(columns={'level_0': 'record_id', 'level_1': 'drop_id', 0: 'new_wifi_infos'}, inplace=True)
    wifi_infos_sparse_matrix_test['wifi_id'] = wifi_infos_sparse_matrix_test['new_wifi_infos'].apply(lambda x: x.split('|')[0])
    wifi_infos_sparse_matrix_test['signal'] = wifi_infos_sparse_matrix_test['new_wifi_infos'].apply(lambda x: int(x.split('|')[1]))
    wifi_infos_sparse_matrix_test.drop(['drop_id', 'new_wifi_infos'], axis=1, inplace=True)
    wifi_infos_sparse_matrix_test = wifi_infos_sparse_matrix_test.groupby(['record_id', 'wifi_id'])['signal'].agg('mean').reset_index()
    wifi_infos_sparse_matrix_test.set_index(['record_id', 'wifi_id'], inplace=True)
    sparse_matrix_test = wifi_infos_sparse_matrix_test.unstack(level = -1)
    len_of_test = len(sparse_matrix_test)



    # merge two dataframes into a dataframe named sparse_matrix by rows(axis=0)
    # in case feature in train and feature in test would miss match
    sparse_matrix = pd.concat([sparse_matrix_train, sparse_matrix_test], axis=0, keys=['train_key', 'test_key'])



    # model is training
    wifi_infos_sparse_matrix_train.reset_index(inplace=True)
    # wifi_infos_sparse_matrix_train.drop(['index'], axis=1, inplace=True)

    wifi_infos_sparse_matrix_train = pd.concat((dataset_for_train, sparse_matrix.loc[:len_of_train]), axis=1)

    test_df = wifi_infos_sparse_matrix_train

    data_for_evaluate = test_df[test_df['time_stamp'] == 0]

    data_for_train = test_df[test_df['time_stamp'] > 0]

    data_for_train_x = data_for_train.drop(['time_stamp', 'shop_label', 'mall_id', 'wifi_infos'], axis=1)
    data_for_train_y = data_for_train[['shop_label']].astype('int')

    data_for_evaluate_x = data_for_evaluate.drop(['time_stamp', 'shop_label', 'mall_id', 'wifi_infos'], axis=1)
    data_for_evaluate_y = data_for_evaluate[['shop_label']].astype('int')

    # save the labelEncoder so as to decode
    labelEncoders.append(temp_le)
    # get and set num_class
    params['num_class'] = len(shop_mall[shop_mall['mall_id'] == mall])

    data_for_train_x.fillna(0, inplace=True)
    data_for_evaluate_x.fillna(0, inplace=True)

    csr = csr_matrix(data_for_train_x.values)

    sparse_predict = csr_matrix(data_for_evaluate_x.values)

    xgb_train_data = xgb.DMatrix(csr, label=data_for_train_y)

    xgb_evaluate_data = xgb.DMatrix(sparse_predict, label=data_for_evaluate_y)

    watchlist = [(xgb_train_data, 'train'), (xgb_evaluate_data, 'evaluate')]
    model = xgb.train(params, xgb_train_data, num_boost_round=500, early_stopping_rounds=20, evals=watchlist)

    # models.append(model)








    wifi_infos_sparse_matrix_test.reset_index(inplace=True)
    wifi_infos_sparse_matrix_test.drop(['index'], axis=1, inplace=True)

    test_dataset_in_this_mall = pd.concat((predict_data, sparse_matrix.loc[len_of_train:len_of_test]), axis=1)

    test_df = test_dataset_in_this_mall

    test_df.drop(['wifi_infos'], axis=1, inplace=True)

    test_df.fillna(0, inplace=True)

    predict_data = csr_matrix(test_df.values)

    xgb_predict_data = xgb.DMatrix(predict_data)
    # get the index of this mall so as to get its model and labelEncoder
    index_of_mall = malls[malls['mall_id'] == mall].index.tolist()[0]
    predict_data['label'] = model.predict(xgb_predict_data)
    predict_data['label'] = predict_data.label.apply(float_to_int)
    predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
    # turn to the main encoder
    predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
    predict_data_id['shop_id'] = predict_data['label']
    result = pd.concat([result, predict_data_id], axis=0)


# predict_malls = evaluation_dataset[['mall_id']]
# predict_malls.drop_duplicates(inplace=True)


# def float_to_int(f):
#     return int(f)


# result = pd.DataFrame()

# for local_mall in predict_malls['mall_id']:
#     print(local_mall)
#     predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == local_mall]
#     predict_data_id = predict_data[['row_id']]
#     # predict_data_mall = predict_data[['mall_id']]
#     predict_data.drop(['user_id', 'time_stamp', 'row_id', 'mall_id'], axis=1, inplace=True)

#     predict_data['wifi_infos_raw'] = predict_data['wifi_infos']

#     # predict_data.drop(['wifi_infos'], axis=1, inplace=True)

#     wifi_infos_sparse_matrix_test = predict_data['wifi_infos'].str.split(";", expand=True).stack().reset_index()
#     wifi_infos_sparse_matrix_test.rename(columns={'level_0': 'record_id', 'level_1': 'drop_id', 0: 'new_wifi_infos'},
#                                          inplace=True)
#     wifi_infos_sparse_matrix_test['wifi_id'] = wifi_infos_sparse_matrix_test['new_wifi_infos'].apply(lambda x: x.split('|')[0])
#     wifi_infos_sparse_matrix_test['signal'] = wifi_infos_sparse_matrix_test['new_wifi_infos'].apply(lambda x: int(x.split('|')[1]))
#     wifi_infos_sparse_matrix_test.drop(['drop_id', 'new_wifi_infos'], axis=1, inplace=True)
#     wifi_infos_sparse_matrix_test = wifi_infos_sparse_matrix_test.groupby(['record_id', 'wifi_id'])['signal'].agg('mean').reset_index()
#     wifi_infos_sparse_matrix_test.set_index(['record_id', 'wifi_id'], inplace=True)
#     sparse_matrix = wifi_infos_sparse_matrix_test.unstack(level = -1)

#     wifi_infos_sparse_matrix_test.reset_index(inplace=True)
#     wifi_infos_sparse_matrix_test.drop(['index'], axis=1, inplace=True)

#     test_dataset_in_this_mall = pd.concat((predict_data, sparse_matrix), axis=1)

#     test_df = test_dataset_in_this_mall

#     test_df.drop(['wifi_infos_raw'], axis=1, inplace=True)

#     test_df.fillna(0, inplace=True)

#     predict_data = csr_matrix(test_df.values)

#     xgb_predict_data = xgb.DMatrix(predict_data)
#     # get the index of this mall so as to get its model and labelEncoder
#     index_of_mall = malls[malls['mall_id'] == local_mall].index.tolist()[0]
#     predict_data['label'] = models[index_of_mall].predict(xgb_predict_data)
#     predict_data['label'] = predict_data.label.apply(float_to_int)
#     predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
#     # turn to the main encoder
#     predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
#     predict_data_id['shop_id'] = predict_data['label']
#     result = pd.concat([result, predict_data_id], axis=0)





    # test_wifi_infos = predict_data[['wifi_infos']]
    # predict_data['wifi_infos'] = test_wifi_infos.wifi_infos.apply(get_best_wifi)
    # predict_data['wifi_infos_1'] = test_wifi_infos.wifi_infos.apply(get_real_best_wifi)
    #
    # xgb_predict_data = xgb.DMatrix(predict_data)
    #
    # index_of_mall = malls[malls['mall_id'] == local_mall].index.tolist()[0]
    # predict_data['label'] = models[index_of_mall].predict(xgb_predict_data)
    # predict_data['label'] = predict_data.label.apply(float_to_int)
    # predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
    #
    # predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
    # predict_data_id['shop_id'] = predict_data['label']
    # result = pd.concat([result, predict_data_id], axis=0)

result.to_csv("result.csv", index=None)
