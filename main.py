from __future__ import print_function
from __future__ import division

                
from pympler import summary
from pympler import muppy
from pympler import tracker


import weakref


#own files
from h5tools import *
from tools import *
from features import *
from settings import *
from classifier import *

import os
import colorama 
colorama.init()
from termcolor import colored
from colorama import Fore, Back, Style
import  fastfilters
import commentjson as json
import pprint
import h5py
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import threading
import multiprocessing





def train(settings):
    



    featuresList = []
    labelsList = []

    for trainingInstanceDataDict in settings.trainignInstancesDataDicts():


        # remember opend h5Files
        openH5Files = []
        
        # get the labels
        f, d = trainingInstanceDataDict['labels']['file']
        labelsDset = h5py.File(f)[d]
        openH5Files.append(f)

        # open all dsets where we need to compute feature on
        dataH5Dsets , openH5Files = settings.getDataH5Dsets(trainingInstanceDataDict, openH5Files)


        # extract the features and the labels
        f, l = extractTrainingData(settings=settings, dataH5Dsets=dataH5Dsets,
                                   labelsH5Dset=labelsDset)
        featuresList.append(f)
        labelsList.append(l)


        closeAllH5Files(openH5Files)

    features = numpy.concatenate(featuresList,axis=0)
    labels = numpy.concatenate(labelsList,  axis=0)

    # substract 1 from labels
    assert labels.min() == 1
    labels -=1

    trainClassifier(settings, features, labels)





def extractTrainingData(settings, dataH5Dsets, labelsH5Dset):

    # check and get shapes
    shape = getShape(dataH5Dsets, labelsH5Dset)


    featuresList = []
    labelsList = []
    lock = threading.Lock()






    outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
    
    numberOfFeatures = 0
    for fOps in outerFeatureOperatorList:
        for fOp in fOps:
            print(fOp,fOp.numberOfFeatures())
            numberOfFeatures += fOp.numberOfFeatures()


    #print("\n\n\n")
    #print("keys",dataH5Dsets.keys())
    #print(maxHaloList)
    ##print(outerFeatureOperatorList[1])
    #sys.exit()

    lockDict = {
    }

    for key in dataH5Dsets.keys():
        lockDict[key] = threading.Lock()


    print("totalshape",shape)

    def f(blockIndex, blockBegin, blockEnd):

        if(settings.useTrainingBlock(blockIndex, blockBegin, blockEnd)):
            labels = loadLabelsBlock(labelsH5Dset, blockBegin, blockEnd)

            if labels.any():
                labels,blockBegin,blockEnd,whereLabels = labelsBoundingBox(labels,blockBegin, blockEnd)
                


                blockShape = [be-bb for be,bb in zip(blockEnd, blockBegin)]



                featureArray = numpy.zeros( (blockShape+[numberOfFeatures]), dtype='float32')

                fIndex = 0 
                for featureOperatorList, maxHalo, dataName in zip(outerFeatureOperatorList, maxHaloList,  dataH5Dsets.keys()):
                    
                    #print("dataName",dataName,featureOperatorList)

                    # the dataset
                    dset = dataH5Dsets[dataName]

                    # add halo to block begin and end
                    gBegin, gEnd ,lBegin, lEnd = addHalo(shape, blockBegin, blockEnd, maxHalo)  



                    slicing = getSlicing(lBegin, lEnd)


                    with lockDict[dataName]:
                        # we load the data with the maximum margin
                        data = loadData(dset, gBegin, gEnd).squeeze()


                    # compute the features
                    for featureOp in featureOperatorList:
                        nf = featureOp.numberOfFeatures()
                        subFeatureArray = featureArray[:,:,:,fIndex:fIndex+nf]

                        

                        fIndex += nf
                        featureOp(data, slicing, subFeatureArray)
         

                labels = labels[whereLabels[0,:],whereLabels[1,:],whereLabels[2,:]]
                with lock:
                    f = featureArray[whereLabels[0,:], whereLabels[1,:], whereLabels[2,:], :]
                    print("appending features:",f.shape)
                    featuresList.append(featureArray[whereLabels[0,:], whereLabels[1,:], whereLabels[2,:], :])
                    labelsList.append(labels)
    

    nWorker = multiprocessing.cpu_count()
    #nWorker = 1
    forEachBlock(shape=shape, blockShape=settings.featureBlockShape,f=f, nWorker=nWorker)





    features = numpy.concatenate(featuresList,axis=0)
    labels = numpy.concatenate(labelsList,  axis=0)
    
    print(numpy.bincount(labels))

    return features,labels

def trainClassifier(settings, features, labels):

    setup = settings.settingsDict['setup']

    nClasses = labels.max() + 1

    clfSetup = setup["classifier"]
    clfType = clfSetup["type"]
    clfSettings = clfSetup["settings"]
    if clfType == "xgb":
        clf = Classifier(nClasses=nClasses, **clfSettings)
        clf.train(X=features, Y=labels)
        
        # save classifer
        clf.save(clfSetup["filename"])
        
    else:
        raise RuntimeError(" %s is a non supported classifer" %clfType)


def loadClassifer(settings):

    clfSetup = settings.settingsDict['setup']["classifier"]
    clfType = clfSetup["type"]
    clfSettings = clfSetup["settings"]
    if clfType == "xgb":
        nt = nWorker = multiprocessing.cpu_count()
        nt = max(1, nt//4)
        nt = 1
        clf = Classifier(nClasses=settings.numberOfClasses,**clfSettings)
        clf.load(clfSetup["filename"], nThreads=nt)
        return clf
    else:
        raise RuntimeError(" %s is a non supported classifer" %clfType)






class PredictionFunctor(object):
    def __init__(self, settings, shape, clf, dataH5Dsets, predictionDset,
        predictionDtype):

        self.settings = settings
        self.shape = shape
        self.clf = clf
        self.dataH5Dsets = dataH5Dsets
        self.predictionDset = predictionDset
        self.predictionDtype = predictionDtype
        outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
        self.outerFeatureOperatorList = outerFeatureOperatorList
        self.maxHaloList = maxHaloList

        self.lock = threading.Lock() 
        self.lockDict = {}

        for key in dataH5Dsets.keys():
            self.lockDict[key] = threading.Lock()


        self.numberOfFeatures = 0
        for fOps in outerFeatureOperatorList:
            for fOp in fOps:
                self.numberOfFeatures += fOp.numberOfFeatures()

        print("numberOfFeatures", self.numberOfFeatures)


    def __call__(self, blockIndex, blockBegin, blockEnd):
        blockShape = (
            blockEnd[0] - blockBegin[0],
            blockEnd[1] - blockBegin[1],
            blockEnd[2] - blockBegin[2]
        )

        with self.lock:
            print("alloc")

        featureArray = numpy.zeros( (blockShape+(self.numberOfFeatures,)), dtype='float32')
        fIndex = 0
  


        for featureOperatorList, maxHalo, dataName in zip(self.outerFeatureOperatorList, self.maxHaloList,  self.dataH5Dsets.keys()):
            
            # the dataset
            dset = self.dataH5Dsets[dataName]

            # add halo to block begin and end
            gBegin, gEnd ,lBegin, lEnd = addHalo(self.shape, blockBegin, blockEnd, maxHalo)  
            slicing = getSlicing(lBegin, lEnd)

            # we load the data with the maximum margin
            with self.lockDict[dataName]:
                data = loadData(dset, gBegin, gEnd).squeeze()

            # compute the features
            for featureOp in featureOperatorList:

                nf = featureOp.numberOfFeatures()
                subFeatureArray = featureArray[:,:,:,fIndex:fIndex+nf]
                fIndex += nf
                featureOp(data, slicing, subFeatureArray)
  


        featuresFlat = featureArray.reshape([-1,self.numberOfFeatures])

        with self.lock:

            #memory_tracker.print_diff()

            #all_objects = muppy.get_objects()
            #sum1 = summary.summarize(all_objects)
            #summary.print_(sum1)      


            probsFlat = self.clf.predict(featuresFlat)
            
            # do the prediction
            #probsFlat = clf.predict(featuresFlat)
            probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
            print("mima",probs.min(),probs.max())
            #print probs
            # convert from float to matching dtype
            if self.predictionDtype == 'uint8':
                probs *= 255.0
                probs = numpy.round(probs,0).astype('uint8')


            self.predictionDset[blockBegin[0]:blockEnd[0],blockBegin[1]:blockEnd[1],blockBegin[2]:blockEnd[2],:] = probs[:,:,:,:]
        


    
def predict(settings):



    # load classifier
    clf = loadClassifer(settings)
    nClasses = settings.numberOfClasses
    




    for predictionInstanceDataDict in settings.predictionInstancesDataDicts(): 
        

        print("pred data dict",predictionInstanceDataDict)

        # remember opend h5Files
        openH5Files = []

        # open all dsets where we need to compute feature on
        dataH5Dsets , openH5Files = settings.getDataH5Dsets(predictionInstanceDataDict, openH5Files)

        # get and check shape
        shape = getShape(dataH5Dsets)


        # allocate output file
        f, d = predictionInstanceDataDict['prediction']['file']

        if os.path.exists(f):
            os.remove(f)

        f = h5py.File(f)
        openH5Files.append(f)
        pshape = shape + (nClasses,)

        predictionDtype = predictionInstanceDataDict['prediction']['dtype']

        predictionDset = f.create_dataset(d,shape=pshape, chunks=(100,100,100,settings.numberOfClasses), dtype=predictionDtype)


        print(predictionDset.dtype)



        

 
        outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
      

        lock = threading.Lock()


        lockDict = {
        }

        for key in dataH5Dsets.keys():
            lockDict[key] = threading.Lock()
        

        #memory_tracker = tracker.SummaryTracker()

        f = PredictionFunctor(settings=settings, shape=shape,
                              clf=clf, dataH5Dsets=dataH5Dsets,
                              predictionDset=predictionDset,
                              predictionDtype=predictionDtype)


        nWorker = multiprocessing.cpu_count()
        print("cpu count",nWorker)
        #nWorker = 1
        #/=


      
        forEachBlock(shape=shape, blockShape=settings.featureBlockShape,f=f, nWorker=nWorker)








        closeAllH5Files(openH5Files)




def importVars(filename, globalVars = None):
    if globalVars is None:
        globalVars = dict()

    localVars = dict()
    with open(filename) as f:
        code = compile(f.read(), filename, 'exec')
        exec(code, globalVars, localVars)
    return localVars


if __name__ == '__main__':
    
    r = 0 

    if r == 0:

        # normaly we use arguments
        settingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/settings.py"
        settingsDict = importVars(settingsFile)['settingsDict']


        if True:
            settings = Settings(settingsDict)
            train(settings=settings)

        if False:
            predictionSettingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/prediction_input.py"   
            predictionSettingsDict = importVars(predictionSettingsFile)['predictionSettingsDict']
            

            settings = Settings(settingsDict, predictionSettingsDict)
            predict(settings=settings)


    else:
        # normaly we use arguments
        settingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/settings_r2.py"
        settingsDict = importVars(settingsFile)['settingsDict']


        if False:
            settings = Settings(settingsDict)
            train(settings=settings)

        if True:
            predictionSettingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/prediction_inputr2.py"   
            predictionSettingsDict = importVars(predictionSettingsFile)['predictionSettingsDict']
            

            settings = Settings(settingsDict, predictionSettingsDict)
            predict(settings=settings)