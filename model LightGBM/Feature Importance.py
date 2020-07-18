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

data = pd.read_csv(r"E:\readmission code\data\simplified HFS ICU data.csv")

# feature_set = [ 'gender', 'age', 'LOS', 'source','severity','diagnosis','disposition', 'temp','resprate', 'hr', 'SpO2','FiO2', 'CVP', 'BPH', 'BPL', 'MAP']

feature_set = [ 'gender', 'age', 'source','diagnosis', 'LOS', 'V1','V2', 'V3','V4', 'V5', 'V6','V7', 'V8', 'V9']

X = data[feature_set]
y = data['readmitted']

from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state=20)
X_train_old, X_test, Y_train_old, y_test = train_test_split(X, y, test_size=0.3,
                                                                random_state=0)
X_train, y_train = smt.fit_sample(X_train_old, Y_train_old)
X_train = pd.DataFrame(X_train, columns=list(X_train_old.columns))
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


y_pred = [list(x).index(max(x)) for x in preds]


acc = accuracy_score(y_test,y_pred)

precision = precision_score(y_test, y_pred, average='macro')

recall = recall_score(y_test, y_pred, average='macro')

sensitivity = sensitivity_score(y_test, y_pred, average='macro')

specificity = specificity_score(y_test, y_pred, average='macro')

f1 = f1_score(y_test, y_pred, average='macro')

gmean = geometric_mean_score(y_test,y_pred, average='macro')

mat = confusion_matrix(y_test, y_pred)

sns.set(font_scale=1)
sns.heatmap(mat, square=True, annot=True, cmap='Reds')
plt.xlabel('True Value')
plt.ylabel('Predict Value')
plt.title('LGBM\n ACC: {0:.2f}%\n F1: {1:.2f}%'.format(acc * 100, f1 * 100))
plt.savefig('./images/LGBM.png')
# plt.show()

feature_names = X_train.columns
feature_imports = gbm.feature_importance()
print(feature_names)
print(feature_imports)
most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)],
                                 columns=["Feature", "Importance"]).nlargest(16, "Importance")

most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(19, 11))
plt.rc('xtick', labelsize=22)
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=22)
plt.xlabel('Importance', fontsize=25)
plt.title('HFS-LightGBM -- Important Features for CVD ICU dataset', fontsize=25)
plt.savefig('./images/LGBMImportance.png')
plt.show()


