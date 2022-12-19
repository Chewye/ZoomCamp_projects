
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv

from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import pickle


df = pd.read_csv("/home/chewye/Документы/data/dataset.csv")



numerical_col = [
    'weight',
    'weight_mfi',
    'price_mfi', 
    'transport_pay',
    'dist_qty_oper_login_1',
    'total_qty_oper_login_1',
    'total_qty_oper_login_0',
    'total_qty_over_index_and_type',
    'total_qty_over_index',
    'label'
]


for col in ['oper_type + oper_attr', 'index_oper', 'type', 'is_privatecategory', 'is_in_yandex', 'is_return']:
    print(f'{col}: {df[df[col]==" "].shape[0]}')


df.loc[(df['index_oper'] == ' '), 'index_oper'] = '0' 
df['index_oper'] = df['index_oper'].astype(np.float64)


df['oper_type'] = df['oper_type + oper_attr'].str.partition('_')[0]
df['oper_attr'] = df['oper_type + oper_attr'].str.partition('_')[2]

df['index_oper'] = df['index_oper'] // 1000


int_col = [
    'index_oper',
    'priority',
    'class',
    'mailtype', 
    'mailctg', 
    'directctg', 
    'postmark',
]

df[int_col] = df[int_col].astype(np.int64)



del_col = ['mailrank', 'id', 'oper_type + oper_attr', 'transport_pay', 'weight_mfi', 'transport_pay']

categorical_col = [
 'index_oper',
 'type',
 'priority', 
 'class',
 'is_privatecategory', 
 'is_in_yandex', 
 'is_return', 
 'mailtype', 
 'mailctg', 
 'directctg', 
 'postmark',
 'is_wrong_sndr_name',
 'is_wrong_rcpn_name',
 'is_wrong_phone_number',
 'is_wrong_address',
 'oper_type',
 'oper_attr'
 ]


df.drop(columns=del_col, axis=1, inplace=True)




train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

x_train = train.drop(['label'], axis=1)
y_train = train['label']

x_test = test.drop(['label'], axis=1)
y_test = test['label']





train_data = Pool(
            x_train,
            label=y_train,
            cat_features=categorical_col,      
            )




params = {'iterations': 200,
          'learning_rate': 0.05,
          'verbose': False,
          'loss_function': 'Logloss',
          'eval_metric': 'AUC',
          'random_seed': 42
          }



score = CatBoostClassifier(**params)
score.fit(train_data, plot=True, eval_set=(x_test, y_test))
score.get_feature_importance(prettified=True)


list_del = list(score.get_feature_importance(prettified=True)['Feature Id'])[13:]




categorical_col = [
 'type',
 'mailctg', 
 'directctg', 
 'is_wrong_rcpn_name',
 'is_wrong_phone_number',
 'oper_type'
  ]
df.drop(columns=list_del ,axis=1, inplace=True)



train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

x_train = train.drop(['label'], axis=1)
y_train = train['label']

x_test = test.drop(['label'], axis=1)
y_test = test['label']



train_data = Pool(
            x_train,
            label=y_train,
            cat_features=categorical_col,      
            )


model = CatBoostClassifier(
    iterations=200,
    random_seed=42,
    learning_rate=0.04,
    eval_metric='AUC',
    l2_leaf_reg=3,
    depth=10
)



model.fit(
    train_data,
    eval_set=(x_test, y_test),
    verbose=False,
    plot=True
)



y_preds = model.predict_proba(x_test)[:,1]

