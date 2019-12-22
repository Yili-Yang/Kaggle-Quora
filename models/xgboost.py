import sys
sys.path.append("..")
from clean.Vectorization.vectorizationMethords import Vectorization
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

vectorization = Vectorization()

#read data
trainDf = pd.read_csv('../data/train.csv')
trainDf['ebd1'] = trainDf.question1.apply(vectorization.word2Vec)
trainDf['ebd2'] = trainDf.question2.apply(vectorization.word2Vec)
#feature building:
x_train = trainDf.apply(lambda x: np.dot(x.ebd1,x.ebd2))

#build training set
y_train = trainDf['is_duplicate']
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

#run model
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)