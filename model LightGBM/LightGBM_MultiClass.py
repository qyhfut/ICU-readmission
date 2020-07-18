import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.metrics import sensitivity_score,specificity_score,geometric_mean_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"E:\readmission code\data\preprocessed ICU data.csv")

# feature_set = [ 'gender', 'age',  'source', 'disposition' ,'diagnosis', 'LOS', 'severity', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

feature_set = [ 'gender', 'age', 'source', 'diagnosis', 'LOS','temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

X = data[feature_set]
y = data['readmitted']

x = np.array(X)
y = np.array(y)

acc_list = []
precision_list = []
recall_list = []
sensitivity_list = []
specificity_list = []
f1_list = []
g_list = []
confusion_list = []

# K-fold cross validation that splits data into train and test set
stra_folder = KFold(n_splits=10, shuffle=True, random_state=42)
# print("y[:10]",y[:10])
for train_index,test_index in stra_folder.split(x,y):
    X_train_old = x[train_index]
    X_test = x[test_index]
    Y_train_old = y[train_index]
    y_test = y[test_index]

    # Xtest = np.array(X_test)
    # ytest = np.array(y_test)
    # test1 = pd.DataFrame(Xtest)
    # test2 = pd.DataFrame(ytest)
    # test1.to_csv("Xtest.csv")
    # test2.to_csv("ytest.csv")

    from imblearn.over_sampling import SMOTE

    smt = SMOTE(random_state=20)
    # X_train_old, X_test, Y_train_old, y_test = train_test_split(X, y, test_size=0.4,
    #                                                             random_state=0)
    X_train, y_train = smt.fit_sample(X_train_old, Y_train_old)
    # X_train = pd.DataFrame(X_train, columns=list(X_train_old.columns))
    print('Original dataset shape %s' % Counter(Y_train_old))
    print('Original dataset shape %s' % Counter(y_train))
    print("nums of train/test set: ", len(X_train), len(X_test), len(y_train), len(y_test))


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_error',
        'max_depth' : -1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'min_sum_hessian_in_leaf' : 0.001,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.5,
    }



    # # XW
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'multiclass',
    #     'num_class': 3,
    # }


    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=500)

    print('Start predicting...')

    preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    y_pred = [list(x).index(max(x)) for x in preds]

    # ypred = np.array(y_pred)
    # test = pd.DataFrame(ypred)
    # test.to_csv("ylabel123.csv")

    acc = accuracy_score(y_test,y_pred)
    acc_list.append(acc)

    precision = precision_score(y_test, y_pred, average='macro')
    precision_list.append(precision)

    recall = recall_score(y_test, y_pred, average='macro')
    recall_list.append(recall)

    sensitivity = sensitivity_score(y_test, y_pred, average='macro')
    sensitivity_list.append(sensitivity)

    specificity = specificity_score(y_test, y_pred, average='macro')
    specificity_list.append(specificity)

    f1 = f1_score(y_test, y_pred, average='macro')
    f1_list.append(f1)

    gmean = geometric_mean_score(y_test,y_pred, average='macro')
    g_list.append(gmean)

    mat = confusion_matrix(y_test, y_pred)
    confusion_list.append(mat)

print("Accuracy: ", acc)
print(acc_list)
print(np.mean(acc_list))

print("Precision:", precision)
print(precision_list)
print(np.mean(precision_list))

print("Recall:", recall)
print(recall_list)
print(np.mean(recall_list))

print("Specificity:",specificity)
print(specificity_list)
print(np.mean(specificity_list))

print("F1 score: ", f1)
print(f1_list)
print(np.mean(f1_list))

print("G_mean:", gmean)
print(g_list)
print(np.mean(g_list))

print("Confusion matrix: \n", mat)

print('Overall report: \n', classification_report(y_test, y_pred))


