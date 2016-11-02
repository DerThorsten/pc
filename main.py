from __future__ import print_function
from __future__ import division


#own files
from h5tools import *
from tools import *
from features import *
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
    
    setup = settings["setup"]

    trainingDataNames = setup['trainingDataNames']
    
    # get all the training data settings
    trainingInstancesSettings = [ ]
    for trainingDataName in trainingDataNames:
        s = settings["trainingData"][trainingDataName]
        s['name'] = trainingDataName
        trainingInstancesSettings.append(s)


    trainImpl(setup=setup, trainingInstancesSettings=trainingInstancesSettings)







def trainImpl(setup, trainingInstancesSettings):



    featuresList = []
    labelsList = []

    for trainingInstanceSettings in trainingInstancesSettings:

        trainingInstanceName = trainingInstanceSettings['name']

        print("  ","trainingInstance:", trainingInstanceName)


        inputFilesSettings = setup["inputFile"]

        # remember opend h5Files
        h5Files = []
        


        f, d = trainingInstanceSettings['labels']['file']
        labelsDset = h5py.File(f)[d]

        h5Files.append(f)

        # combine all the input files h5 files
        # (like raw data, or predictions from prev.rounds)
        dataH5Dsets = OrderedDict()
        for inputFileSettings in inputFilesSettings:
            inputFileName = inputFileSettings['name']
            print("    ","inputFile:",inputFileName)
        
            # get the h5filename 
            f,d = trainingInstanceSettings['data'][inputFileName]['file']

            h5File = h5py.File(f,'r')
            dset = h5File[d]

            # dsets
            dataH5Dsets[inputFileName] = dset

            # remeber all files opend
            h5Files.append(h5File)


        # extract the features and the labels
        f, l = extractTrainingData(setup=setup, dataH5Dsets=dataH5Dsets,
                                   labelsH5Dset=labelsDset)
        featuresList.append(f)
        labelsList.append(l)


        closeAllH5Files(h5Files)

    features = numpy.concatenate(featuresList,axis=0)
    labels = numpy.concatenate(labelsList,  axis=0)

    # substract 1 from labels
    assert labels.min() == 1
    labels -=1

    trainClassifier(setup, features, labels)

def extractTrainingData(setup, dataH5Dsets, labelsH5Dset):

    # check shapes
    shape = tuple(labelsH5Dset.shape[0:3])
    for inputFileName in dataH5Dsets.keys():
        fshape = tuple(dataH5Dsets[inputFileName].shape[0:3])
        assert shape == fshape

    blockShape = tuple(setup['blockShape'])



    featuresList = []
    labelsList = []
    lock = threading.Lock()

    futures = []
    nWorker = multiprocessing.cpu_count()
    #nWorker = 1


    rawFeatureExtractor = RawFeatureExtractor()
    rawFeatureExtractor.shape = shape

    with ThreadPoolExecutor(max_workers=nWorker) as executer:
        for blockBegin, blockEnd in blockYielder((0,0,0), shape, blockShape):
            
            
            @reraise_with_stack
            def f(blockBegin, blockEnd):
                #print(blockBegin,blockEnd)
                
                labels = loadLabelsBlock(labelsH5Dset, blockBegin, blockEnd)
                if labels.any():
                    #   print("hasLabels")

                    labels,blockBegin,blockEnd,whereLabels = labelsBoundingBox(labels,blockBegin, blockEnd)
                    features = []
                    for dsNames in dataH5Dsets.keys():
                        features.append(rawFeatureExtractor(dataH5Dsets[dsNames], blockBegin, blockEnd, whereLabels))

                    features = numpy.concatenate(features,axis=1)
                    labels = labels[whereLabels[0,:],whereLabels[1,:],whereLabels[2,:]]
                    with lock:
                        print("appending features:",features.shape)
                        featuresList.append(features)
                        labelsList.append(labels)

            if nWorker == 1:
                f(blockBegin, blockEnd)
            else:
                future = executer.submit(f, blockBegin, blockEnd)
                futures.append(future)


    for future in futures:
        e = future.exception()
        if e is not None:
            raise e



    features = numpy.concatenate(featuresList,axis=0)
    labels = numpy.concatenate(labelsList,  axis=0)
    
    print(numpy.bincount(labels))

    return features,labels



def trainClassifier(setup, features, labels):

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


def loadClassifer(settings, predictionSettings):

    clfSetup = settings['setup']["classifier"]
    clfType = clfSetup["type"]
    clfSettings = clfSetup["settings"]
    if clfType == "xgb":
        clf = Classifier(nClasses=settings['setup']['nClasses'],**clfSettings)
        clf.load(clfSetup["filename"])#, nThreads=None)
        return clf
    else:
        raise RuntimeError(" %s is a non supported classifer" %clfType)


    
def predict(settings, predictionSettings):

    setup = settings["setup"]
    predictionInput = predictionSettings["predictionInput"]


    # load classifier
    clf = loadClassifer(settings, predictionSettings)


    nClasses = setup['nClasses']



    for dataName in predictionInput.keys(): 
        


        inputFilesSettings = setup["inputFile"]

        # remember opend h5Files
        h5Files = []
        
        shape = None

        # combine all the input files h5 files
        # (like raw data, or predictions from prev.rounds)
        dataH5Dsets = OrderedDict()
        for inputFileSettings in inputFilesSettings:
            inputFileName = inputFileSettings['name']
            print("    ","inputFile:",inputFileName)
        
            # get the h5filename 
            f,d = predictionInput[dataName]['data'][inputFileName]['file']

            h5File = h5py.File(f,'r')
            dset = h5File[d]

            if shape is None:
                shape = tuple(dset.shape[0:3])
            else:
                assert shape == tuple(dset.shape[0:3])

            # dsets
            dataH5Dsets[inputFileName] = dset


        # allocate output file
        f, d = predictionInput[dataName]['prediction']['file']

        if os.path.exists(f):
            os.remove(f)

        f = h5py.File(f)
        h5Files.append(f)
        pshape = shape + (nClasses,)
        predictionDset = f.create_dataset(d,shape=pshape, chunks=(100,100,100,1), dtype='float32')


        nWorker = multiprocessing.cpu_count()
        #nWorker = 1

        rawFeatureExtractor = RawFeatureExtractor()
        rawFeatureExtractor.shape = shape
        blockShape = predictionSettings['setup']['blockShape']

        futures = []

        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=nWorker) as executer:
            for blockBegin, blockEnd in blockYielder((0,0,0), shape, blockShape):
                
                
                assert blockEnd[0] <= shape[0]
                assert blockEnd[1] <= shape[1]
                assert blockEnd[2] <= shape[2]
                @reraise_with_stack
                def f(blockBegin, blockEnd):

                    blockShape = tuple([e-b for e,b in zip(blockEnd, blockBegin)])

                    features = []
                    for dsNames in dataH5Dsets.keys():
                        features.append(rawFeatureExtractor(dataH5Dsets[dsNames], blockBegin, blockEnd))

                    features = numpy.concatenate(features,axis=3)

                    nFeatures = features.shape[3]
                    featuresFlat = features.reshape([-1,nFeatures])

                    with lock:
                        probs = clf.predict(featuresFlat)
                        probs = probs.reshape(blockShape+(-1,))
                        predictionDset[
                            blockBegin[0]:blockEnd[0], 
                            blockBegin[1]:blockEnd[1], 
                            blockBegin[2]:blockEnd[2],
                            :
                        ] = probs
                    #print(probs)

                if nWorker == 1:
                    f(blockBegin, blockEnd)
                else:
                    future = executer.submit(f, blockBegin, blockEnd)
                    futures.append(future)


        for future in futures:
            e = future.exception()
            if e is not None:
                raise e





            # remeber all files opend
            h5Files.append(h5File)




        closeAllH5Files(h5Files)








if __name__ == '__main__':
    

    # normaly we use arguments
    settingsFile = "/home/tbeier/src/neurocontext/data/hhess_supersmall/settings.json"

    
    # read settings file
    with open(settingsFile) as jsonFile:
        jsonStr = jsonFile.read()
        settings = json.loads(jsonStr)


    if True:

        train(settings=settings)

    if True:
        predictionSettingsFile = "/home/tbeier/src/neurocontext/data/hhess_supersmall/prediction_input.json"   

        # read settings file
        with open(predictionSettingsFile) as jsonFile:
            jsonStr = jsonFile.read()
            predictionSettings = json.loads(jsonStr)

        predict(settings=settings, predictionSettings=predictionSettings)
