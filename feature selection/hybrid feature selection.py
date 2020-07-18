#HFS algorithm combing filter and wrapper methods
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold,cross_val_predict
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def correlation_filter(datamatrix):
    df_all_data = datamatrix
    corr_matrix = df_all_data.iloc[:,0:(df_all_data.shape[1]-1)].corr()
    cormat = []
    for i in range(len(corr_matrix)):
        f1 = corr_matrix.columns[i]
        for j in range(i,len(corr_matrix)):
            f2 = corr_matrix.columns[j]
            cormat.append([f1, f2, corr_matrix.iloc[i,j]])
    cormat = pd.DataFrame(cormat,columns=['f1','f2','values'])
    # cormat.head(5)
    cormat_filter = cormat.loc[(cormat['values']>=0.9) & (cormat['values'] !=1.0)]
    todrop = set(cormat_filter['f2'])
    df_all_data.drop(todrop, axis=1, inplace=True)

    print ("After Correlation filter >=0.9: Removed " + str(len(todrop)) + " features from the dataset")
    return df_all_data

def get_feature_ranking(X_train,y_train):

    model = lgb.LGBMClassifier()
    rfecv = RFECV(estimator=model, step=1, min_features_to_select = 15, cv=KFold(10), scoring='accuracy')
    rfecv = rfecv.fit(X_train, y_train)
    # print(rfecv.n_features_)
    # print(rfecv.support_)
    # print(rfecv.ranking_)
    lgb_ranking = []
    for x,w in zip(rfecv.ranking_, X_train.columns):
       lgb_ranking.append([w, x])
    lgb_ranking = pd.DataFrame(lgb_ranking, columns=['features', 'score'])
    lgb_ranking.sort_values('features', inplace=True)

    df_ranked = lgb_ranking

    return df_ranked


def get_best_features(df_all_data):

    print ("correlation filtering....")
    df_all_data_1 = correlation_filter(df_all_data)

    X_train = df_all_data_1.iloc[:, 0:(df_all_data.shape[1] - 1)]
    y_train = df_all_data_1.iloc[:, (df_all_data.shape[1] - 1)]

    print ("feature ranking started....")
    df_ranked = get_feature_ranking(X_train,y_train)
    # df_ranked.columns = ['ranked_features', 'score']
    df_ranked.to_csv("features_ranking10.csv",index=False)


#read file
df_all_data = pd.read_csv(r"E:\readmission code\data\preprocessed ICU data.csv")
get_best_features(df_all_data)
