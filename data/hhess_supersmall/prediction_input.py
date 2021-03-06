predictionSettingsDict = {

    "predictionInput" : {
        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/home/tbeier/src/pc/data/hhess_supersmall/raw.h5",
                        "dataa"
                    ]
                }
            },
            "prediction" :{
                "file" : [
                    "/home/tbeier/src/pc/data/hhess_supersmall/raw_predictions.h5",
                    "data"
                ],
                # must be either 'float32','float64', or 'uint8'
                "dtype" : "uint8"
            }
        }   
    }
    ,
    "setup" : {   
        "blockShape" : [60,60,60]
    }
}