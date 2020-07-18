import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"E:\readmission code\preprocessed ICU data.csv")

feature_set = [ 'gender', 'age','source', 'diagnosis', 'LOS', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

# feature_set = [ 'gender', 'age',  'source', 'disposition' ,'diagnosis', 'LOS', 'severity', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

X = data[feature_set]
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


#Set initial parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.1
}

data_train = lgb.Dataset(X_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=10, stratified=False, shuffle=True, metrics='multi_logloss',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['multi_logloss-mean']))
print('best cv score:', cv_results['multi_logloss-mean'][-1])


#search max_depth and num_leaves
from sklearn.model_selection import GridSearchCV

model_lgb = lgb.LGBMClassifier(objective = 'multiclass',num_class = 3, metric = 'multi_logloss',
                             n_estimators = 15, learning_rate = 0.1)

params_test1={
    'max_depth': range(-1,3,2),
    'num_leaves':range(31,70,50)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_log_loss', cv=10, verbose=1, n_jobs=4)

gsearch1.fit(X_train, y_train)
print(gsearch1.best_params_, gsearch1.best_score_)


#set min_data_in_leaf and min_sum_hessian_in_leaf
params_test3={
    'min_data_in_leaf': [10,11,12,13,14,15,16,17,18,19,20],
    'min_sum_hessian_in_leaf':[0.001, 0.002]
}

model_lgb = lgb.LGBMClassifier(objective = 'multiclass', num_class = 3, metric = 'multi_logloss',
                               n_estimators = 15, max_depth = -1, num_leaves = 31,learning_rate = 0.1)

gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_log_loss', cv=10, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
print(gsearch3.best_params_, gsearch2.best_score_)


#set feature_fraction and bagging_fraction
params_test4={
    'feature_fraction': [ 0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [ 0.6, 0.7, 0.8, 0.9, 1.0]
}

model_lgb = lgb.LGBMClassifier(objective = 'multiclass',num_class = 3, metric = 'multi_logloss',
                               n_estimators = 15,  max_depth = -1, num_leaves = 31,
                               min_data_in_leaf = 20, min_sum_hessian_in_leaf = 0.001,
                               learning_rate = 0.1)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_log_loss', cv=10, verbose=1, n_jobs=4)
gsearch4.fit(X_train, y_train)
print(gsearch4.best_params_, gsearch3.best_score_)


