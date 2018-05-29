#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:42:24 2017

@author: lab-tan.yun
"""

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

shop_info_raw = pd.read_csv('data/ccf_first_round_shop_info.csv')
user_info_raw = pd.read_csv('data/ccf_first_round_user_shop_behavior.csv')
evaluation_dataset = pd.read_csv('data/evaluation_public.csv')
#get mall_id
malls = shop_info_raw[['mall_id']]
malls.drop_duplicates(inplace = True)
malls.reset_index(inplace = True)
malls = malls.drop(['index'],axis = 1)

#get shop label
mainLabelEncoder = LabelEncoder()
shop_label = shop_info_raw[['shop_id']]
mainLabelEncoder.fit(shop_label)

user_in_shop = user_info_raw[['shop_id']]
user_in_shop = mainLabelEncoder.transform(user_in_shop)

user_info_labeled = user_info_raw
user_info_labeled['shop_label'] = user_in_shop

#user_info_labeled = user_info_labeled.drop(['shop_id','user_id','time_stamp'],axis = 1)
user_info_labeled = user_info_labeled.drop(['shop_id','user_id'],axis = 1)

def divide_data_into_four_part(s):
   day = int(s.split(' ')[0].split('-')[2])%4
   if day == 0:
       return 0
   elif day == 1:
       return 1
   elif day == 2:
       return 2
   elif day == 3:
       return 3
    
user_info_labeled['time_stamp'] = user_info_labeled.time_stamp.apply(divide_data_into_four_part)


#divide the data into two part
user_info_eval = user_info_labeled[user_info_labeled['time_stamp'] == 0]
user_info_train = user_info_labeled[user_info_labeled['time_stamp'] > 0]


#as i think, if we only use the best one, it may be over fitting
def get_best_wifi(s):
    s = s.split(';')
    best_wifi = ''
    strength = -999
    for i in range(len(s)):
        w = s[i].split('|')
        if  int(w[1]) > strength:
            best_wifi = w[0]
            strength = int(w[1])
    best_wifi = best_wifi.split('_')
    return int(best_wifi[1])

#unfinished,if the wifi signal are too close or even they have the same strength.
#the trouble will come.
#im not sure if this will cause over fitting.
#Stop doing this , try the simplest one.
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
    
    #we suppose to get the closet group.
    #but wait, we should  do the simple one first.
    temp_best_id = []
    temp_second_id = []
    for i in range(len(temp_s)):
        for j in range(len(temp_best)):
            if temp_s[i].find(str(temp_best[j])) > -1:
                temp_best_id.append(int(temp_s[i].split('|')[0].split('_')[1]))
                
        for j in range(len(temp_second)):
            if temp_s[i].find(str(temp_second[j])) > -1:
                temp_second_id.append(int(temp_s[i].split('|')[0].split('_')[1]))
    #now the best and the second id all get
    
    temp_best_id.sort(reverse = True)
    temp_second_id.sort(reverse = True)
    
#    best_wifi = best_wifi.split('_')
#    if second_wifi != 'null':
#        second_wifi = second_wifi.split('_')
#        return int(best_wifi[1]),int(second_wifi[1])
    #keep relax....Think about the wonderful life.
    return temp_best_id[0],temp_second_id[0]



user_info_labeled['wifi_infos'] = user_info_labeled.wifi_infos.apply(get_best_wifi)

shop_mall = shop_info_raw[['shop_id','mall_id']]
shop_mall['shop_id'] = mainLabelEncoder.fit_transform(shop_info_raw['shop_id'])
shop_mall.rename(columns={'shop_id':'shop_label'},inplace = True)


user_info_labeled = pd.merge(user_info_labeled,shop_mall,on=['shop_label'],how='left')


#at here,we are going to train our models
#we need different encoder for different mall

#the common parameter for all models
#exclude num_class,becasue this parameter is unique in each mall
params={'booster':'gbtree',
	    'objective': 'multi:softmax',
	    'eval_metric':'merror',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':6,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.1,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

models = []
labelEncoders = []
for mall in malls['mall_id']:
    print(mall)
    #prepare data in this mall
    dataset_for_train = user_info_labeled[user_info_labeled['mall_id'] == mall]
    dataset_for_train_x = dataset_for_train[['longitude', 'latitude', 'wifi_infos']]
    dataset_for_train_y = dataset_for_train[['shop_label']]
    #init shop labelEncoder for this mall
    temp_le = LabelEncoder()
    dataset_for_train_y = temp_le.fit_transform(dataset_for_train_y['shop_label'])
    #save the labelEncoder so as to decode
    labelEncoders.append(temp_le)
    #get and set num_class 
    params['num_class'] = len(shop_mall[shop_mall['mall_id'] == mall])
    
    xgb_data = xgb.DMatrix(dataset_for_train_x,label=dataset_for_train_y)
    watchlist = [(xgb_data,'train')]
    model = xgb.train(params,xgb_data,num_boost_round=50,evals=watchlist)
    models.append(model)

#the mall_ids are given
#handle the predict data

predict_malls = evaluation_dataset[['mall_id']]
predict_malls.drop_duplicates(inplace = True)

def float_to_int(f):
    return int(f)


result = pd.DataFrame()

for local_mall in predict_malls['mall_id']:
    print(local_mall)
    predict_data = evaluation_dataset[evaluation_dataset['mall_id'] == local_mall]
    predict_data_id = predict_data[['row_id']]
    #predict_data_mall = predict_data[['mall_id']]
    predict_data.drop(['user_id','time_stamp','row_id','mall_id'],axis = 1,inplace=True)
    predict_data['wifi_infos'] = predict_data.wifi_infos.apply(get_best_wifi)
    
    xgb_predict_data = xgb.DMatrix(predict_data)
    #get the index of this mall so as to get its model and labelEncoder
    index_of_mall = malls[malls['mall_id'] == local_mall].index.tolist()[0]
    predict_data['label'] = models[index_of_mall].predict(xgb_predict_data)
    predict_data['label'] = predict_data.label.apply(float_to_int)
    predict_data['label'] = labelEncoders[index_of_mall].inverse_transform(predict_data['label'])
    #turn to the main encoder
    predict_data['label'] = mainLabelEncoder.inverse_transform(predict_data['label'])
    predict_data_id['shop_id'] = predict_data['label']
    result = pd.concat([result,predict_data_id],axis = 0)

result.to_csv("result.csv",index=None)
