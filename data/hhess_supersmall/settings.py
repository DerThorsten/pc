from __future__ import print_function
from __future__ import division



def useBlock(blockIndex, blockBegin, blockEnd):
    #print("bi",blockIndex)
    return True#(blockIndex + 1) % 2 == 0


settingsDict = {
    "trainingData" : {

        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/home/tbeier/src/pc/data/hhess_supersmall/raw.h5",
                        "dataa"
                    ]
                }
            }
            ,
            "labels" : {

                "file" : [
                    "/home/tbeier/src/pc/data/hhess_supersmall/explicit_semantic_labels.h5",
                    "data"
                ]
            }
        }   
    },

    "setup" : {   
        "useBlock" : useBlock,
        "blockShape" : [50,50,50],
        "nClasses" : 7,
        "trainingDataNames": [
            "hhess"
        ],
        "featureSettings" : [
            {
                "name" : "raw",
                "features" :[
                    {
                        "type" : "ConvolutionFeatures",
                        "kwargs" : {
                            "sigmas" : [1.0,2.0,4.0,8.0,12.0]
                        }
                    }
                ]
            }
        ] 
        ,
        "classifier" :{
            "training_set": "/home/tbeier/src/pc/data/hhess_supersmall/training_set_r1.h5",
            "filename":"/home/tbeier/src/pc/data/hhess_supersmall/clf.clf",
            "type" : "rf",
            "settings" : {
                "n_jobs":-1, 
                "n_estimators":255
            }
        }
    }
}