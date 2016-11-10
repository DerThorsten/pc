predictionSettingsDict = {

    "predictionInput" : {
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
            },
            "prediction" :{
                "file" : [
                    "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_semantic_r2_full.h5",
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