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
                        "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/data_normalized.h5",
                        "data"
                    ]
                }
            }
            ,
            "labels" : {

                "file" : [
                    "/home/tbeier/src/pc/data/hhess_full_2nm/labels_semantic.h5",
                    "exported_data"
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
                            "sigmas" : [1.0,2.0,4.0,8.0,12.0]
                        }
                    }
                ]
            }
        ] 
        ,
        "classifier" :{
            "training_set": "/home/tbeier/src/pc/data/hhess_full_2nm/training_set_r1.h5",
            "filename":"/home/tbeier/src/pc/data/hhess_full_2nm/clf.clf",
            "type" : "rf",
            "settings" : {
                "n_jobs":-1, 
                "n_estimators":255
            }
        }
    }
}