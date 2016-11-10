from __future__ import print_function
from __future__ import division

                
# from pympler import summary
# from pympler import muppy
# from pympler import tracker


import weakref
import progressbar
from progressbar import *
#own files
from h5tools import *
from tools import *
from features import *
from settings import *
from classifier import *

import pylab
import vigra
import os
import colorama 
colorama.init()
from termcolor import colored,cprint
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


    nBlocks = 0
    for blockIndex, blockBegin, blockEnd in blockYielder((0,0,0), shape, settings.featureBlockShape):
        nBlocks +=1

    widgets = ['Training: ', Percentage(), ' ',Counter(),'/',str(nBlocks),'', Bar(marker='0',left='[',right=']'),
           ' ', ETA()] #see docs for other options
                       #
    bar = progressbar.ProgressBar(maxval=nBlocks,widgets=widgets)
    doneBlocks = [0]
    bar.start()
    def f(blockIndex, blockBegin, blockEnd):

        if(settings.useTrainingBlock(blockIndex, blockBegin, blockEnd)):
            labels = loadLabelsBlock(labelsH5Dset, blockBegin, blockEnd)

            #with lock:
            #    print("np unique",numpy.unique(labels))

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
                    #print("appending features:",f.shape)
                    featuresList.append(featureArray[whereLabels[0,:], whereLabels[1,:], whereLabels[2,:], :])
                    labelsList.append(labels)

        with lock:
            #print(doneBlocks)
            doneBlocks[0] += 1
            bar.update(doneBlocks[0])
    

    nWorker = multiprocessing.cpu_count()
    #nWorker = 1
    forEachBlock(shape=shape, blockShape=settings.featureBlockShape,f=f, nWorker=nWorker)


    bar.finish()


    features = numpy.concatenate(featuresList,axis=0)
    labels = numpy.concatenate(labelsList,  axis=0)
    
    print(numpy.bincount(labels))

    return features,labels

def trainClassifier(settings, features, labels):
    print("train classifier")
    setup = settings.settingsDict['setup']
    f = setup["classifier"]["training_set"]

    if os.path.exists(f):
        os.remove(f)



    h5file = h5py.File(f,'w')
    h5file['features'] = features
    h5file['labels'] = labels
    h5file.close()


   

    nClasses = labels.max() + 1

    clfSetup = setup["classifier"]
    clfType = clfSetup["type"]
    clfSettings = clfSetup["settings"]
    if clfType == "xgb":
        clf = XGBClassifier(nClasses=nClasses, **clfSettings)
        clf.train(X=features, Y=labels)
        
        # save classifer
        clf.save(clfSetup["filename"])

    elif clfType == "rf":
        clf = RfClassifier(**clfSettings)
        clf.train(X=features, Y=labels)
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
        clf = XGBClassifier(nClasses=settings.numberOfClasses,**clfSettings)
        clf.load(clfSetup["filename"], nThreads=5)
        return clf
    elif clfType == "rf":
        clf = RfClassifier(**clfSettings)
        clf.load(clfSetup["filename"], nThreads=1)
        return clf

    else:
        raise RuntimeError(" %s is a non supported classifer" %clfType)






class PredictionFunctor(object):
    def __init__(self, settings, shape, clf, dataH5Dsets, predictionDset,
        predictionDtype, roiBegin, roiEnd):

        self.settings = settings
        self.shape = shape
        self.clf = clf
        self.dataH5Dsets = dataH5Dsets
        self.predictionDset = predictionDset
        self.predictionDtype = predictionDtype
        self.roiBegin = roiBegin
        self.roiEnd = roiEnd
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
            for i,featureOp in enumerate(featureOperatorList):

                


                nf = featureOp.numberOfFeatures()
                subFeatureArray = featureArray[:,:,:,fIndex:fIndex+nf]
                fIndex += nf
                featureOp(data, slicing, subFeatureArray)
  
                #if(i==1 and dataName=='pmap'):
                #    for x in range(subFeatureArray.shape[3]):
                #        p = int(subFeatureArray.shape[2]/2)
                #        fImg = subFeatureArray[:,:,p,x]
                #        print(data.shape)
                #        rImg = data[slicing+[slice(0,1)]][:,:,p,0]
                #        f = pylab.figure()
                #        f.add_subplot(2, 1, 1)
                #        pylab.imshow(fImg,cmap='gray')
                #        f.add_subplot(2, 1, 2)
                #        pylab.imshow(rImg,cmap='gray')
                #        pylab.show()

        featuresFlat = featureArray.reshape([-1,self.numberOfFeatures])


        if self.clf.needsLockedPrediction():




            with self.lock:
                probsFlat = self.clf.predict(featuresFlat)
                probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
                print("mima",probs.min(),probs.max())
                if self.predictionDtype == 'uint8':
                    probs *= 255.0
                    probs = numpy.round(probs,0).astype('uint8')


                self.predictionDset[blockBegin[0]:blockEnd[0],blockBegin[1]:blockEnd[1],blockBegin[2]:blockEnd[2],:] = probs[:,:,:,:]
            
        else:

            probsFlat = self.clf.predict(featuresFlat)
            probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
            if self.predictionDtype == 'uint8':
                probs *= 255.0
                probs = numpy.round(probs,0).astype('uint8')

            with self.lock:
                print("mima",probs.min(),probs.max())
                dsetBegin = [bb-rb for bb,rb in zip(blockBegin, self.roiBegin)]
                dsetEnd = [be-rb for bb,rb in zip(blockEnd, self.roiBegin)]
                self.predictionDset[dsetBegin[0]:dsetEnd[0],dsetBegin[1]:dsetEnd[1],dsetBegin[2]:dsetEnd[2],:] = probs[:,:,:,:]

    
def predict(settings):



    with Timer("load classifier:"):
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
        roiBegin, roiEnd = predictionInstanceDataDict['prediction'].get('roi',[[0,0,0],shape])
        roiShape = [re-rb for re,rb in zip(roiEnd, roiBegin)]

        f, d = predictionInstanceDataDict['prediction']['file']

        if os.path.exists(f):
            os.remove(f)

        f = h5py.File(f)
        openH5Files.append(f)
        pshape = roiShape + [nClasses]

        predictionDtype = predictionInstanceDataDict['prediction']['dtype']


        chunkShape = tuple([min(s,c) for s,c in zip(pshape[0:3],(100,100,100))]) + (settings.numberOfClasses,)

        predictionDset = f.create_dataset(d,shape=pshape, chunks=chunkShape, dtype=predictionDtype)





        

 
        outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
      

        lock = threading.Lock()


        lockDict = {
        }

        for key in dataH5Dsets.keys():
            lockDict[key] = threading.Lock()
        

        outerFeatureOperatorList, maxHaloList = settings.getFeatureOperators()
       

        numberOfFeatures = 0
        for fOps in outerFeatureOperatorList:
            for fOp in fOps:
                numberOfFeatures += fOp.numberOfFeatures()


        nBlocks = 0
        for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, settings.featureBlockShape):
            nBlocks +=1


        widgets = ['Prediction: ', Percentage(), ' ',Counter(),'/',str(nBlocks), Bar(marker='0',left='[',right=']'),
               ' ', ETA()] #see docs for other options
                           #
        bar = progressbar.ProgressBar(maxval=nBlocks,widgets=widgets)
        doneBlocks = [0]
        bar.start()




        def  f(blockIndex, blockBegin, blockEnd):
            blockShape = (
                blockEnd[0] - blockBegin[0],
                blockEnd[1] - blockBegin[1],
                blockEnd[2] - blockBegin[2]
            )

            #with lock:
            #    print("alloc")

            featureArray = numpy.zeros( (blockShape+(numberOfFeatures,)), dtype='float32')
            fIndex = 0
      


            for featureOperatorList, maxHalo, dataName in zip(outerFeatureOperatorList, maxHaloList,  dataH5Dsets.keys()):
                
                # the dataset
                dset = dataH5Dsets[dataName]

                # add halo to block begin and end
                gBegin, gEnd ,lBegin, lEnd = addHalo(shape, blockBegin, blockEnd, maxHalo)  
                slicing = getSlicing(lBegin, lEnd)

                # we load the data with the maximum margin
                with lockDict[dataName]:
                    data = loadData(dset, gBegin, gEnd).squeeze()

                # compute the features
                for i,featureOp in enumerate(featureOperatorList):

                    


                    nf = featureOp.numberOfFeatures()
                    subFeatureArray = featureArray[:,:,:,fIndex:fIndex+nf]
                    fIndex += nf
                    featureOp(data, slicing, subFeatureArray)
      

            featuresFlat = featureArray.reshape([-1,numberOfFeatures])


            if clf.needsLockedPrediction():




                with lock:

                    doneBlocks[0] += 1
                    bar.update(doneBlocks[0])

                    probsFlat = clf.predict(featuresFlat)
                    probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
                    #print("mima",probs.min(),probs.max())
                    if predictionDtype == 'uint8':
                        probs *= 255.0
                        probs = numpy.round(probs,0).astype('uint8')


                    dsetBegin = [bb-rb for bb,rb in zip(blockBegin, roiBegin)]
                    dsetEnd = [be-rb for be,rb in zip(blockEnd, roiBegin)]
                    predictionDset[dsetBegin[0]:dsetEnd[0],dsetBegin[1]:dsetEnd[1],dsetBegin[2]:dsetEnd[2],:] = probs[:,:,:,:]

                
            else:
                doneBlocks[0] += 1
                bar.update(doneBlocks[0])

                probsFlat = clf.predict(featuresFlat)
                probs = probsFlat.reshape(tuple(blockShape)+(settings.numberOfClasses,))
                if predictionDtype == 'uint8':
                    probs *= 255.0
                    probs = numpy.round(probs,0).astype('uint8')

                with lock:
                    #print("mima",probs.min(),probs.max())
                    dsetBegin = [bb-rb for bb,rb in zip(blockBegin, roiBegin)]
                    dsetEnd = [be-rb for be,rb in zip(blockEnd, roiBegin)]
                    predictionDset[dsetBegin[0]:dsetEnd[0],dsetBegin[1]:dsetEnd[1],dsetBegin[2]:dsetEnd[2],:] = probs[:,:,:,:]

    



        nWorker = multiprocessing.cpu_count()
        #nWorker = 1
        #/=




      
        forEachBlock(shape=shape, roiBegin=roiBegin, roiEnd=roiEnd, blockShape=settings.featureBlockShape,f=f, nWorker=nWorker)

        bar.finish()






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
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str,choices=['train','predict'],
                        help="train or predict")

    parser.add_argument('settings', nargs='*', default=os.getcwd())
    args = parser.parse_args()




    






    # normaly we use arguments
    settingsFile = args.settings[0]
    settingsDict = importVars(settingsFile)['settingsDict']


    if args.mode=='train':
        print("TRAINING:")
        settings = Settings(settingsDict)
        train(settings=settings)

    elif args.mode == 'predict':
        print("PREDICTION:")
        if len(args.settings) != 2:
            parser.error('if mode == predict a valid prediction_settings filename is needed')

        predictionSettingsFile = args.settings[1]   
        predictionSettingsDict = importVars(predictionSettingsFile)['predictionSettingsDict']
        

        settings = Settings(settingsDict, predictionSettingsDict)
        predict(settings=settings)
