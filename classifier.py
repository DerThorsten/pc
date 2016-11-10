import xgboost as xgb
import multiprocessing
from sklearn.ensemble import RandomForestClassifier

import cPickle

class XGBClassifier(object):
    def __init__(self, nClasses, nRounds=200, maxDepth=3, nThreads=multiprocessing.cpu_count(), silent=1):
        self.param = {
            'bst:max_depth':maxDepth, 
            'bst:eta':1, 
            'silent':silent, 
            'num_class':nClasses,
            'objective':'multi:softprob' 
        }
        self.param['nthread'] = nThreads
        
        self.nRounds = nRounds 
        self.nThreads = nThreads

    def train(self, X, Y, getApproxError=False):
        assert self.param['num_class'] is not None

        dtrain = xgb.DMatrix(X, label=Y)
        self.bst = xgb.train(self.param, dtrain, self.nRounds)

        if getApproxError:

            e = 0.0
            c = 0.0

            kf = KFold(Y.shape[0], n_folds=4)
            for train_index, test_index in kf:

                XTrain = X[train_index, :]
                XTest  = X[test_index, :]

                YTrain = Y[train_index]
                YTest  = Y[test_index]

                dtrain2 = xgb.DMatrix(XTrain, label=YTrain)
                bst = xgb.train(self.param, dtrain2, self.nRounds)
              

                dtest = xgb.DMatrix(XTest)
                probs = bst.predict(dtest)
                ypred =numpy.argmax(probs, axis=1)

                

                error = float(numpy.sum(ypred != YTest))
                e += error
                c += float(len(YTest))

            e/=c

            return e

    def needsLockedPrediction(self):
        return True



    def save(self, fname):
        self.bst.save_model(fname)

    def load(self, fname, nThreads=None):
        if nThreads is None:
            self.bst = xgb.Booster({'nthread':self.nThreads})
        else:
            self.bst = xgb.Booster({'nthread':nThreads})

        self.bst.load_model(fname)            


    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)





class RfClassifier():
    def __init__(self,**kwargs):
        self.params = kwargs
        self.clf  = None




    def train(self, X, Y):
        if self.clf is None:
            self.clf = RandomForestClassifier(**self.params)

        self.clf.fit(X,Y)


    def save(self, fname):
        assert  self.clf is not None
        with open(fname, 'wb') as fid:
            cPickle.dump(self.clf, fid)    

    def load(self, fname, nThreads=None):
        with open(fname, 'rb') as fid:
            self.clf = cPickle.load(fid)

        if nThreads is not None:
            self.params['n_jobs'] = nThreads

        self.clf.set_params(**self.params)
            


    def predict(self, X):
        assert self.clf is not None
        return self.clf.predict_proba(X)


    def needsLockedPrediction(self):
        return False
