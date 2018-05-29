import pandas as pd
import numpy as np
from datetime import date
shop_info = pd.read_csv('data/ccf_first_round_shop_info_train.csv')
train = pd.read_csv('data/ccf_first_round_user_shop_behavior_train.csv')

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
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 
    return c * r * 1000 
shop_info_0 = shop_info[['mall_id']]
shop_info_0['shop_num'] = 1               ##加shop_num
shop_info_0 = shop_info_0.groupby(['mall_id']).agg('sum').reset_index()
shop_info_1 = shop_info[['mall_id','longitude','latitude']] #加mall经纬度
shop_info_1 = shop_info_1.groupby(['mall_id']).agg('mean').reset_index()
shop_info_1.rename(columns={'longitude': 'mall_longitude', 'latitude': 'mall_latitude'}, inplace=True) 
shop_info = shop_info.merge(shop_info_0, on = 'mall_id', how = 'left')
shop_info = shop_info.merge(shop_info_1, on = 'mall_id', how = 'left')
shop_info['s_m_distance'] = list(map(lambda x1, y1, x2, y2: haversine(x1, y1 , x2, y2) , shop_info['longitude'], shop_info['latitude'], shop_info['mall_longitude'], shop_info['mall_latitude']))

from datetime import datetime
from dateutil.parser import parse
from datetime import date
from datetime import time
train['time_stamp'] = train['time_stamp'].apply(parse)
train['date'] = train['time_stamp'].apply(datetime.date)
train['time'] = train['time_stamp'].apply(datetime.time)
train['day_of_week'] = train.date.apply(lambda x:date.weekday(x)+1)
train['is_weekend'] = train.day_of_week.apply(lambda x:1 if x in (6,7) else 0)

def get_hour(t):
    h,m,s = t.split(':')
    return h
train['hour'] = train['time'].apply(str).apply(get_hour)

def get_day(t):
    y,m,d = t.split('-')
    return d
train['day'] = train['date'].apply(str).apply(get_day)

def is_connect(s):
    wifi_list = s.split(';')
    is_connect = 0
    for i in wifi_list:
        wifi_id,power,connect = i.split('|')
        if connect=='true':
            is_connect = wifi_id.split('_')[1]
    return is_connect
def most_power_id(s):
    wifi_list = s.split(';')
    wifi_dict = {}
    for i in wifi_list:
        wifi_id,power,connect = i.split('|')
        power = float(power)
        wifi_dict[wifi_id] = power
    wifi_sort=sorted(wifi_dict.items(),key = lambda x:x[1],reverse = True)
    results=[key for key,value in wifi_sort]  
    if len(results)>=3:    
        return results[0]+";"+results[1]+";"+results[2]
    else:
        for i in range(len(results),3):
            results.append('b_0')
        return results[0]+";"+results[1]+";"+results[2]
train['is_connect'] = train.wifi_infos.astype('str').apply(is_connect)
train['most_power_id'] = train.wifi_infos.astype('str').apply(most_power_id)
train['most_power_wifi_0'] = train.most_power_id.apply(lambda x:x.split(';')[0])
train['most_power_wifi_1'] = train.most_power_id.apply(lambda x:x.split(';')[1])
train['most_power_wifi_2'] = train.most_power_id.apply(lambda x:x.split(';')[2])
train['most_power_wifi_0'] = train.most_power_wifi_0.apply(lambda x:x.split('_')[1])
train['most_power_wifi_1'] = train.most_power_wifi_1.apply(lambda x:x.split('_')[1])
train['most_power_wifi_2'] = train.most_power_wifi_2.apply(lambda x:x.split('_')[1])

train = train.reset_index()
train.rename(columns={'index':'idx'},inplace=True)
shop_mall = shop_info[['shop_id','mall_id']]
train = train.merge(shop_mall,on = 'shop_id',how = 'left')
temp = train[['idx','mall_id']]
temp = temp.merge(shop_mall,on = 'mall_id',how = 'left')
temp = temp.sample(n=2000000,axis = 0)
temp.drop(['mall_id'],axis=1,inplace=True)
temp = train.merge(temp,on = 'idx',how = 'right')
temp['label'] = list(map(lambda x, y: 1 if(x==y) else 0,temp['shop_id_x'], temp['shop_id_y']))
temp = temp[temp.label == 0]
temp = temp.reset_index(drop=True)
temp.drop(['shop_id_x'],axis=1,inplace=True)
temp.rename(columns={'shop_id_y':'shop_id'},inplace=True)

train['label'] = 1
train = train.append(temp).reset_index(drop = True)
train = train.merge(shop_info,on = 'shop_id',how = 'left')
train.drop(['mall_id_x'],axis=1,inplace=True)
train.rename(columns={'mall_id_y':'mall_id'},inplace=True)
train['u_s_distance'] = list(map(lambda x1, y1, x2, y2: haversine(x1, y1 , x2, y2) , train['longitude_x'], train['latitude_x'], train['longitude_y'], train['latitude_y']))
train.drop(['user_id','time_stamp','wifi_infos','date','time','most_power_id'],axis=1,inplace=True)
train['shop_id'] = train.shop_id.apply(lambda x:x.split('_')[1])
train['mall_id'] = train.mall_id.apply(lambda x:x.split('_')[1])
train['category_id'] = train.category_id.apply(lambda x:x.split('_')[1])
train.day = train.day.astype('int')
train.hour = train.hour.astype('int')
train.most_power_wifi_0 = train.most_power_wifi_0.astype('int')
train.most_power_wifi_1 = train.most_power_wifi_1.astype('int')
train.most_power_wifi_2 = train.most_power_wifi_2.astype('int')
train.shop_id = train.shop_id.astype('int')
train.mall_id = train.mall_id.astype('int')
train.category_id = train.category_id.astype('int')
train.drop(['idx'],axis=1,inplace=True)
train.to_csv('data/processed_train_data.csv',index=None)
temp = train.loc[[0], :]
temp.to_csv('data/columns_seq.csv',index=None)