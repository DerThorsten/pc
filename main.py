from __future__ import print_function
from __future__ import division


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
    
    #print("\n\n\n")
    #print("keys",dataH5Dsets.keys())
    #print(maxHaloList)
    ##print(outerFeatureOperatorList[1])
    #sys.exit()

    lockDict = {
    }

    for key in dataH5Dsets.keys():
        lockDict[key] = threading.Lock()


    @reraise_with_stack
    def f(blockBegin, blockEnd):
        labels = loadLabelsBlock(labelsH5Dset, blockBegin, blockEnd)

        if labels.any():
            labels,blockBegin,blockEnd,whereLabels = labelsBoundingBox(labels,blockBegin, blockEnd)



            features = []

            for featureOperatorList, maxHalo, dataName in zip(outerFeatureOperatorList, maxHaloList,  dataH5Dsets.keys()):
                
                #print("dataName",dataName,featureOperatorList)

                # the dataset
                dset = dataH5Dsets[dataName]

                # add halo to block begin and end
                gBegin, gEnd ,lBegin, lEnd = addHalo(shape, blockBegin, blockEnd, maxHalo)  

                with lockDict[dataName]:
                    # we load the data with the maximum margin
                    data = loadData(dset, gBegin, gEnd).squeeze()

                # compute the features
                for featureOp in featureOperatorList:
                    feat = featureOp(data)[whereLabels[0,:] + lBegin[0], whereLabels[1,:] + lBegin[1], whereLabels[2,:] + lBegin[2], :]
                    #with lock:
                    #    print("subfeat ",feat.shape)
                    features.append(feat)

            #with lock:
            #    print("flist ",len(features))        
            features = numpy.concatenate(features,axis=1)
            labels = labels[whereLabels[0,:],whereLabels[1,:],whereLabels[2,:]]
            with lock:
                print("appending features:",features.shape)
                featuresList.append(features)
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
        clf = Classifier(nClasses=settings.numberOfClasses,**clfSettings)
        clf.load(clfSetup["filename"])#, nThreads=None)
        return clf
    else:
        raise RuntimeError(" %s is a non supported classifer" %clfType)


    
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





        

 
        outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
      

        lock = threading.Lock()


        lockDict = {
        }

        for key in dataH5Dsets.keys():
            lockDict[key] = threading.Lock()
        @reraise_with_stack
        def f(blockBegin, blockEnd):
            
            blockShape = (
                blockEnd[0] - blockBegin[0],
                blockEnd[1] - blockBegin[1],
                blockEnd[2] - blockBegin[2]
            )
      
            features = []

            for featureOperatorList, maxHalo, dataName in zip(outerFeatureOperatorList, maxHaloList,  dataH5Dsets.keys()):
                
                # the dataset
                dset = dataH5Dsets[dataName]

                # add halo to block begin and end
                gBegin, gEnd ,lBegin, lEnd = addHalo(shape, blockBegin, blockEnd, maxHalo)  

                # we load the data with the maximum margin
                with lockDict[dataName]:
                    data = loadData(dset, gBegin, gEnd).squeeze()

                # compute the features
                for featureOp in featureOperatorList:
                    feat = featureOp(data)[lBegin[0]:lEnd[0],lBegin[1]:lEnd[1],lBegin[2]:lEnd[2],:]
                    features.append(feat)

            features = numpy.concatenate(features,axis=3)
            nFeatures = features.shape[3]

            featuresFlat = features.reshape([-1,nFeatures])

            with lock:
                # do the prediction
                probsFlat = clf.predict(featuresFlat)
                probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
                print("mima",probs.min(),probs.max())
                #print probs
                # convert from float to matching dtype
                if predictionDtype == 'uint8':
                    probs *= 255.0
                    probs = numpy.round(probs,0).astype('uint8')

                predictionDset[blockBegin[0]:blockEnd[0],blockBegin[1]:blockEnd[1],blockBegin[2]:blockEnd[2],:] = probs[:,:,:,:]


        nWorker = multiprocessing.cpu_count()
        #nWorker = 1
        forEachBlock(shape=shape, blockShape=settings.featureBlockShape,f=f, nWorker=nWorker)








        closeAllH5Files(openH5Files)








if __name__ == '__main__':
    

    # normaly we use arguments
    settingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/settingsr2.json"

    
    # read settings file
    with open(settingsFile) as jsonFile:
        jsonStr = jsonFile.read()
        settingsDict = json.loads(jsonStr)

    settings = Settings(settingsDict)

    if False:
        
        train(settings=settings)

    if True:
        predictionSettingsFile = "/home/tbeier/src/pc/data/hhess_supersmall/prediction_inputr2.json"   

        # read settings file
        with open(predictionSettingsFile) as jsonFile:
            jsonStr = jsonFile.read()
            predictionSettingsDict = json.loads(jsonStr)

        settings = Settings(settingsDict, predictionSettingsDict)
        predict(settings=settings)
