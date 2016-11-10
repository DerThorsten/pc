predictionSettingsDict = {

    "predictionInput" : {
        "hhess" : {
            "data":{
                "raw" : {
                    "file" : [
                        "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/data_normalized.h5",
                        "data"
                    ]
                }
            },
            "prediction" :{
                "file" : [
                    "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_semantic_full.h5",
                    "data"
                ],
                "roi": [[0,0,0], [2000,2000,2000]],
                "dtype" : "uint8"
            }
        }   
    }
    ,
    "setup" : {   
        "blockShape" : [64,64,64]
    }
}