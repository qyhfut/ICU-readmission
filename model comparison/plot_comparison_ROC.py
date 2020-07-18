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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r"E:\readmission code\data\preprocessed ICU data.csv")

feature_set = [ 'gender', 'age',  'LOS','source', 'severity','diagnosis','disposition' ,'temp', 'resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

X = data[feature_set]
y = data['readmitted']

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=20)
X_train_old, X_test, Y_train_old, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, y_train = smt.fit_sample(X_train_old, Y_train_old)
X_train = pd.DataFrame(X_train, columns=list(X_train_old.columns))
print('Original dataset shape %s' % Counter(Y_train_old))
print("nums of train/test set: ", len(X_train), len(X_test), len(y_train), len(y_test))


# from sklearn.ensemble import RandomForestClassifier
# print('--- Random-forest model ---')
# forest = RandomForestClassifier(random_state=20)
# forest.fit(X_train, y_train)
# y_score = forest.predict(X_test)
# y_proba = forest.predict_proba(X_test)


# from sklearn.ensemble import ExtraTreesClassifier
# print('--- Extratrees model ---')
# extra = ExtraTreesClassifier(random_state=20)
# extra.fit(X_train, y_train)
# y_score = extra.predict(X_test)
# y_proba = extra.predict_proba(X_test)


# from sklearn.ensemble import AdaBoostClassifier
# print('--- AdaBoost model ---')
# adaboost = AdaBoostClassifier(random_state=20)
# adaboost.fit(X_train, y_train)
# y_score = adaboost.predict(X_test)
# y_proba = adaboost.predict_proba(X_test)


# from sklearn.ensemble import GradientBoostingClassifier
# print('--- GradientBoosting model ---')
# gradient = GradientBoostingClassifier(random_state=20)
# gradient.fit(X_train, y_train)
# y_score = gradient.predict(X_test)
# y_proba = gradient.predict_proba(X_test)


from xgboost import XGBClassifier
import xgboost as xgb
print('--- XGBoost model ---')
model = xgb.XGBClassifier(random_state=20)
model.fit(X_train, y_train)
y_score = model.predict(X_test)
y_proba = model.predict_proba(X_test)


n_classes = 3
# # Binarize the output
y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
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
plt.title('Receiver operating characteristic of RF')
plt.legend(loc="lower right")
plt.show()


allfpr = np.array(fpr["macro"])
test = pd.DataFrame(allfpr)
test.to_csv("XGBoostfpr.csv")

meantpr = np.array(tpr["macro"])
test1 = pd.DataFrame(meantpr)
test1.to_csv("XGBoosttpr.csv")


plt.plot(fpr[0], tpr[0], color='r', label='ROC curve of class 1 (area = {1:0.2f})'
                                          ''.format(0, roc_auc[0]), linewidth=1, alpha=0.6)
test2 = pd.DataFrame({"fpr":fpr[0],"tpr":tpr[0]})
test2.to_csv("XGBoost0.csv")
print(fpr[0])
print(tpr[0])

plt.plot(fpr[1], tpr[1], color='green', label='ROC curve of class 2 (area = {1:0.2f})'
                                              ''.format(1, roc_auc[1]), linewidth=1, alpha=0.6)

test3 = pd.DataFrame({"fpr":fpr[1],"tpr":tpr[1]})
test3.to_csv("XGBoost1.csv")
print(fpr[0])
print(tpr[0])

plt.plot(fpr[2], tpr[2], color='blue', label='ROC curve of class 3 (area = {1:0.2f})'
                                             ''.format(2, roc_auc[2]), linewidth=1, alpha=0.6)

test4 = pd.DataFrame({"fpr":fpr[2],"tpr":tpr[2]})
test4.to_csv("XGBoost2.csv")


# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class of RF')
plt.legend(loc="lower right")
plt.show()




