from __future__ import print_function
from __future__ import division



def useBlock(blockIndex, blockBegin, blockEnd):
    #print("bi",blockIndex)
    return True#(blockIndex + 1) % 2 != 0


settingsDict = {
    "trainingData" : {

        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/home/tbeier/raw_sub.h5",
                        "data"
                    ]
                },
                "pmap" : {
                    "file" : [
                        "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_semantic_full.h5",
                        "data"
                    ]
                }
            }
            ,
            "labels" : {

                "file" : [
                    "/home/tbeier/src/pc/data/hhess_full_2nm/labels_semantic_r2.h5",
                    "data"
                ]
            }
        }   
    },

    "setup" : {   
        "useBlock" : useBlock,
        "blockShape" : [60,60,60],
        "nClasses" : 3,
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
                            "sigmas" : [1.0,2.0,4.0, 8.0, 12.0], 
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
                            "sigmas" : [1.0, 2.0, 4.0, 8.0, 12],
                            "usedChannels" : [0, 1, 2, 3 ,4 ,5,6]
                        }
                    },
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 0,
                    #        "thresholds":[128.0]
                    #    }
                    #},
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 1,
                    #        "thresholds":[128.0]
                    #    }
                    #},
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 2,
                    #        "thresholds":[128.0]
                    #    }
                    #},
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 3,
                    #        "thresholds":[128.0]
                    #    }
                    #},
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 4,
                    #        "thresholds":[128.0]
                    #    }
                    #},
                    #{
                    #    "type" : "BinaryMorphologyFeatures",
                    #    "kwargs" : {
                    #        "channel" : 5,
                    #        "thresholds":[128.0]
                    #    }
                    #}
                ]
            }
        ] 
        ,
        "classifier" :{
            "training_set": "/home/tbeier/src/pc/data/hhess_full_2nm/training_set_r2.h5",
            "filename":"/home/tbeier/src/pc/data/hhess_full_2nm/clfr2.clf",

            "type" : "rf",
            "settings" : {
                "n_jobs":-1, 
                "n_estimators":255
            }
        }
    }
}