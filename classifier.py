import xgboost as xgb
import multiprocessing

class Classifier(object):
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