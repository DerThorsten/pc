from __future__ import print_function
from __future__ import division



def useBlock(blockIndex, blockBegin, blockEnd):
    #print("bi",blockIndex)
    return (blockIndex + 1) % 2 != 0


settingsDict = {
    "trainingData" : {

        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/home/tbeier/src/pc/data/hhess_supersmall/raw.h5",
                        "data"
                    ]
                },
                "pmap" : {
                    "file" : [
                        "/home/tbeier/src/pc/data/hhess_supersmall/raw_predictions.h5",
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
        "blockShape" : [64,64,64],
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
                            "sigmas" : [1.0,2.0,4.0, 8.0], 
                        }
                    }
                ]
            },
            {
                "name" : "pmap",
                "features" :[
                    {
                        "type" : "ConvolutionFeatures",
                        "kwargs" : {
                            "sigmas" : [2.0, 4.0, 8.0, 12],
                            "usedChannels" : [0, 1, 2, 3, 4, 5, 6]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 0,
                            "thresholds":[128.0]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 1,
                            "thresholds":[128.0]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 2,
                            "thresholds":[128.0]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 3,
                            "thresholds":[128.0]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 4,
                            "thresholds":[128.0]
                        }
                    },
                    {
                        "type" : "BinaryMorphologyFeatures",
                        "kwargs" : {
                            "channel" : 5,
                            "thresholds":[128.0]
                        }
                    }
                ]
            }
        ] 
        ,
        "classifier" :{
            "filename":"/home/tbeier/src/pc/data/hhess_supersmall/clfr2.clf",

            "type" : "xgb",
            "settings" : {
                "nRounds":25, 
                "maxDepth":3
            }
        }
    }
}