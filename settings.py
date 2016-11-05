from collections import OrderedDict
import h5py
from features import registerdFeatureOperators
class Settings(object):
    def __init__(self, settingsDict, predictionSettingsDict=None):

        self.settingsDict  = settingsDict

        self.featureBlockShape =  tuple(self.settingsDict["setup"]["blockShape"])
        
        if predictionSettingsDict is not None:
            self.featureBlockShape = tuple(predictionSettingsDict['setup']["blockShape"])


        self.numberOfClasses = self.settingsDict["setup"]["nClasses"]
        self.predictionSettingsDict = predictionSettingsDict


    def trainingInstancesNames(self):

        setup = self.settingsDict["setup"]
        return setup['trainingDataNames']

    def predictionInstancesNames(self):
        
        return self.predictionSettingsDict['predictionInput'].keys()

    def trainignInstancesDataDicts(self):
        setup = self.settingsDict["setup"]
        trainingDataNames = setup['trainingDataNames']
        trainingInstancesSettings = [ ]

        for trainingDataName in trainingDataNames:
            s = self.settingsDict["trainingData"][trainingDataName]
            s['name'] = trainingDataName
            trainingInstancesSettings.append(s)
        return trainingInstancesSettings


    def predictionInstancesDataDicts(self):
        assert self.predictionSettingsDict is not None
        d =  self.predictionSettingsDict['predictionInput']

        dicts = []

        for key in d.keys():
            ddict = d[key]
            ddict['name'] = key
            dicts.append(ddict)
        return dicts



    def featureSetttingsList(self):
        return self.settingsDict["setup"]["featureSettings"]


    def getLabelsH5Path(self, instanceName):
        trainingInstanceDataDict = self.settingsDict["trainingData"][instanceName]
        f,d = trainingInstanceDataDict['labels']
        return f,d

    def getDataH5Dsets(self, instanceDataDict, openH5Files):

        dataH5Dsets = OrderedDict()
        for featureSettings in self.featureSetttingsList():
            inputFileName = featureSettings['name']
            print("    ","inputFile:",inputFileName)
        
            # get the h5filename 
            dataDict = instanceDataDict['data']
            f,d = dataDict[inputFileName]['file']

            h5File = h5py.File(f,'r')
            dset = h5File[d]



            # dsets
            dataH5Dsets[inputFileName] = dset

            # remeber all files opend
            openH5Files.append(h5File)

        return dataH5Dsets, openH5Files



    def getFeatureOperators(self):

        dataH5Dsets = OrderedDict()
        
        outerList = []
        maxHaloList = []

        #print("fs0",self.featureSetttingsList()[0])
        #print("fs1",self.featureSetttingsList()[0])
        for featureSettings in self.featureSetttingsList():

            inputFileName = featureSettings['name']

            #print("features for",inputFileName)
            
            featureOperatorsSettingsList = featureSettings["features"]

            innerList = []

            maxHalo = (0,0,0)

            for featureOperatorSettings in featureOperatorsSettingsList:
                #print(featureOperatorSettings)
                
                fOpName = featureOperatorSettings['type']
                fOpKwargs = featureOperatorSettings['kwargs']
                fOpCls = registerdFeatureOperators[fOpName]
                fOp = fOpCls(**fOpKwargs)

                halo = fOp.halo()

                
                maxHalo = map(lambda aa,bb: max(aa,bb), halo, maxHalo)

                innerList.append(fOp)

            outerList.append(innerList)

            maxHaloList.append(maxHalo)
        
        return outerList,maxHaloList

