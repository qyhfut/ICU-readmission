import numpy as np
import pandas as pd
import scipy.stats as sp

# file path
DATA_DIR = "./data"
ORI_DATA_PATH = DATA_DIR + "/ICU data.csv"
# MAP_PATH = DATA_DIR + "/IDs_mapping.csv"
OUTPUT_DATA_PATH = DATA_DIR + "/preprocessed ICU data.csv"


# load data
dataframe_ori = pd.read_csv(ORI_DATA_PATH)
NUM_RECORDS = dataframe_ori.shape[0]
NUM_FEATURE = dataframe_ori.shape[1]
print(NUM_RECORDS)
print(NUM_FEATURE)

# make a copy of the dataframe for preprocessing
df = dataframe_ori.copy(deep=True)


# encode gender
df['gender'] = df['gender'].replace('男', 1)
df['gender'] = df['gender'].replace('女', 0)


#encode source
df['source'] = df['source'].replace('耳鼻喉科',0)
df['source'] = df['source'].replace('急诊内科',1)
df['source'] = df['source'].replace('急诊外科',2)
df['source'] = df['source'].replace('神经外科',3)
df['source'] = df['source'].replace('心内科',4)
df['source'] = df['source'].replace('心脏大血管外科',5)
df['source'] = df['source'].replace('血管甲状腺外科',6)
df['source'] = df['source'].replace('其他',7)

#encode disposition
df['disposition'] = df['disposition'].replace('XW ICU',0)

#encode diagnosis
df['diagnosis'] = df['diagnosis'].replace('传染病',0)
df['diagnosis'] = df['diagnosis'].replace('创伤',1)
df['diagnosis'] = df['diagnosis'].replace('肺部疾病',2)
df['diagnosis'] = df['diagnosis'].replace('循环系统疾病',3)
df['diagnosis'] = df['diagnosis'].replace('其他',4)

#encode severity
df['severity'] = df['severity'].replace('一般',0)
df['severity'] = df['severity'].replace('病重',1)

def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0))


numerics = ['age', 'LOS', 'temp', 'resprate', 'hr', 'SpO2', 'FiO2', 'CVP', 'BPH',
            'BPL', 'MAP']


df[numerics] = standardize(df[numerics])
# df = df[(np.abs(sp.stats.zscore(df[numerics])) < 5).all(axis=1)]



print('begin out')
print(OUTPUT_DATA_PATH)
df.to_csv(OUTPUT_DATA_PATH)
