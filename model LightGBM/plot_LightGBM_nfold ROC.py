import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import csv

data = pd.read_csv(r"E:\readmission code\data\preprocessed ICU data.csv")

feature_set = [ 'gender', 'age',  'source', 'disposition' ,'diagnosis', 'LOS', 'severity', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

# feature_set = [ 'gender', 'age',  'source', 'diagnosis', 'LOS', 'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']


X = data[feature_set]
y = data['readmitted']

x = np.array(X)
y = np.array(y)

tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)
j=1

# csvfile='HFS-LightGBM_class_roc.csv'
#
# write = open(csvfile, 'a', newline='',encoding='gb18030')
#
# writer = csv.writer(write)

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
    # print('Original dataset shape %s' % Counter(Y_train_old))
    # print('Original dataset shape %s' % Counter(y_train))
    print("nums of train/test set: ", len(X_train), len(X_test), len(y_train), len(y_test))

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'multiclass',
    #     'num_class': 3,
    #     'metric': 'multi_error',
    #     'max_depth' : -1,
    #     'num_leaves': 31,
    #     'min_data_in_leaf': 20,
    #     'min_sum_hessian_in_leaf' : 0.001,
    #     'learning_rate': 0.01,
    #     'feature_fraction': 0.7,
    #     'bagging_fraction': 0.6,
    # }


    # XW
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
    }


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

    n_classes = 3
    # # Binarize the output
    y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        recording=[]
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # recording.append(fpr[i])
        # recording.append(tpr[i])
        # recording.append(roc_auc[i])
        # writer.writerow(recording)

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

    aucs.append(roc_auc["macro"])

    tprs.append(interp(mean_fpr, fpr["macro"], tpr["macro"]))
    tprs[-1][0] = 0.0

    plt.plot(fpr["macro"], tpr["macro"], lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (j, roc_auc["macro"]))
    j += 1


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(tprs, axis=0)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of LightGBM')
plt.legend(loc='lower right')
plt.show()


allfpr = np.array(mean_fpr)
test = pd.DataFrame(allfpr)
test.to_csv("LightGBM-mean_fpr.csv")

meantpr = np.array(mean_tpr)
test1 = pd.DataFrame(meantpr)
test1.to_csv("LightGBM-mean_tpr.csv")


