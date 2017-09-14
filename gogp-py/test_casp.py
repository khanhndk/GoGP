from gogp_s import GOGP_S
from util import Util as U
import pickle
import sklearn.preprocessing as skpre
from sklearn.datasets import load_svmlight_file

datadir = '../datasets/'
dataset = 'casp.shuffle_2'
filename = datadir+dataset+'.txt'

xxTrain, yyTrain = load_svmlight_file(filename)
xxTrain = xxTrain.toarray()
yyTrain = yyTrain.reshape(yyTrain.shape[0], 1)

# print xxTrain.shape
# print yyTrain.shape

# min_max_scaler = skpre.MinMaxScaler(feature_range=(-1, 1))
# xxTrain = min_max_scaler.fit_transform(xxTrain)

learner = GOGP_S(theta=0.6585, gamma=0.02, lbd=0.004, percent_batch=0.1)
learner.fit_online_delay(xxTrain, yyTrain)
print 'RMSE (Online):', learner.final_rmse
print 'Training time:', learner.online_time

print 'save report ...'
learner.X = None
pickle.dump(learner, open("log/" + dataset + U.get_string_time('.log.p'), "wb" ))
