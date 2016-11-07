predictionSettingsDict = {

    "predictionInput" : {
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
            },
            "prediction" :{
                "file" : [
                    "/home/tbeier/src/pc/data/hhess_supersmall/pred_r2.h5",
                    "data"
                ],
                # must be either 'float32','float64', or 'uint8'
                "dtype" : "uint8"
            }
        }   
    }
    ,
    "setup" : {   
        "blockShape" : [50,50,50]
    }
}