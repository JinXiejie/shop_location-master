import xgboost as xgb
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train = pd.read_csv('D:/TencentAds/TencentLookalike/MetaData/train_extract.csv')
test = pd.read_csv('D:/TencentAds/TencentLookalike/MetaData/test_extract.csv')
print len(train)
df = train[train['label'] == 1]
print len(df)
train = train[train['label'] == -1]
print len(train)
train = train.apply(lambda t: t.sample(int(len(t) * 0.05), axis=0, random_state=1))
print len(train)
train = pd.concat((train, df))


def logistic_label(label):
    if label == -1:
        return 0
    return label


train['label'] = train['label'].apply(logistic_label)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.05,
    'max_depth': 9,
    'eval_metric': 'auc',
    'seed': 1024,
    # 'missing': -999,
    'silent': 1,
    'subsample': 0.7,
    # 'scale_pos_weight': 4
}

feature = [x for x in train.columns if
           x not in ['label', 'marriageStatus', 'interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2',
                     'ct', 'os']]

X_train, X_test, y_train, y_test = train_test_split(train[feature], train['label'].astype('int'),
                                                    test_size=0.2, random_state=0)

# xgb_scaler = StandardScaler()
# xgb_scaler.fit(X_train)
# X_train = xgb_scaler.transform(X_train)
# X_test = xgb_scaler.transform(X_test)

xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

# xgbtrain = xgb.DMatrix(train[feature], train['label'])
# xgbeval = xgb.DMatrix(train[feature], train['label'])
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=270, early_stopping_rounds=50, evals=watchlist)
model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)
# 267

xgbtest = xgb.DMatrix(test[feature])
result = pd.DataFrame()
result['score'] = model.predict(xgbtest)
result = pd.concat((test['aid, uid'], result), axis=1)
result.to_csv('D:/TencentAds/TencentLookalike/Code/submission_csv/submission.csv', index=None)

train = train.fillna(train.median())
X_train, X_test, y_train, y_test = train_test_split(train[feature], train['label'].astype('int'),
                                                    test_size=0.2, random_state=0)
logistic_model = LogisticRegression(penalty='l2', fit_intercept=True, max_iter=300)
logistic_model.fit(X_train, y_train)
logistic_predict = logistic_model.predict(X_test)

