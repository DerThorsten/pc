from __future__ import print_function
from __future__ import division



def useBlock(blockIndex, blockBegin, blockEnd):
    #print("bi",blockIndex)
    return (blockIndex + 1) % 2 == 0


settingsDict = {
    "trainingData" : {

        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/home/tbeier/src/pc/data/hhess_supersmall/raw.h5",
                        "data"
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
        "blockShape" : [150,150,150],
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
                            "sigmas" : [1.0,2.0,4.0,8.0]
                        }
                    }
                ]
            }
        ] 
        ,
        "classifier" :{
            "filename":"/home/tbeier/src/pc/data/hhess_supersmall/clf.clf",

            "type" : "xgb",
            "settings" : {
                "nRounds":100, 
                "maxDepth":3
            }
        }
    }
}