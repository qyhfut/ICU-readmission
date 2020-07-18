import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"E:\readmission code\data\preprocessed ICU data.csv")

# feature_set = [ 'gender', 'age',  'source', 'disposition' ,'diagnosis', 'LOS', 'severity', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

feature_set = [ 'gender', 'age',  'source', 'diagnosis', 'LOS', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

X = data[feature_set]
y = data['readmitted']

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=20)
X_train_old, X_test, Y_train_old, y_test = train_test_split(X, y, test_size=0.3,random_state=0)#train_test_split()是随机划分训练集
X_train, y_train = smt.fit_sample(X_train_old,Y_train_old)
X_train = pd.DataFrame(X_train, columns=list(X_train_old.columns))
print(X_train_old)
print(X_train)
print('Original dataset shape %s' % Counter(Y_train_old))
print('Original dataset shape %s' % Counter(y_train))
print("nums of train/test set: ", len(X_train), len(X_test), len(y_train), len(y_test))

import lightgbm as lgb
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
        'bagging_fraction': 0.6,
    }


# params = {
#         'boosting_type': 'gbdt',
#         'objective': 'multiclass',
#         'num_class': 3,
#     }


# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Start predicting...')

preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)


n_classes = 3
# # Binarize the output
y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), preds.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linewidth=1, alpha=0.6)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linewidth=1, alpha=0.6)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of LightGBM')
plt.legend(loc="lower right")
plt.show()


import pandas as pd
allfpr = np.array(fpr["macro"])
test = pd.DataFrame(allfpr)
test.to_csv("HFS-LightGBMfpr.csv")

meantpr = np.array(tpr["macro"])
test1 = pd.DataFrame(meantpr)
test1.to_csv("HFS-LightGBMtpr.csv")


plt.plot(fpr[0], tpr[0], color='r', label='ROC curve of class 1 (area = {1:0.2f})'
                                          ''.format(0, roc_auc[0]), linewidth=1, alpha=0.6)
test2 = pd.DataFrame({"fpr":fpr[0],"tpr":tpr[0]})
test2.to_csv("HFS-LightGBM0.csv")
print(fpr[0])
print(tpr[0])

plt.plot(fpr[1], tpr[1], color='green', label='ROC curve of class 2 (area = {1:0.2f})'
                                              ''.format(1, roc_auc[1]), linewidth=1, alpha=0.6)

test3 = pd.DataFrame({"fpr":fpr[1],"tpr":tpr[1]})
test3.to_csv("HFS-LightGBM1.csv")
print(fpr[0])
print(tpr[0])

plt.plot(fpr[2], tpr[2], color='blue', label='ROC curve of class 3 (area = {1:0.2f})'
                                             ''.format(2, roc_auc[2]), linewidth=1, alpha=0.6)

test4 = pd.DataFrame({"fpr":fpr[2],"tpr":tpr[2]})
test4.to_csv("HFS-LightGBM2.csv")


# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class of LightGBM')
plt.legend(loc="lower right")
plt.show()

