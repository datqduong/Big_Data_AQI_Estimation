# Multi-source Machine Learning for AQI Estimation
This repository contains the source code for the IEEE Big Data 2020 Conference: 6th Special Session on Intelligent Data Mining paper [Multi-source Machine Learning for AQI Estimation]() by Dat Q. Duong, Quang M. Le, Tan-Loc Nguyen-Tai, Dong Bo, Dat Nguyen, Minh-Son Dao, and Binh T. Nguyen.

## Data sets

<table> 
    <tr> 
        <td> <b>MNR-HCM</b> </td>
        <td> <b>MNR-Air HCM</b> </td>
        <td> <b>SEPHLA - MediaEval 2019</b></td>
    </tr>
    <tr>
      <td valign="top"><img src="Screenshots/mnr-route-min.png" width=400 height=300"></td>
      <td valign="top"><img src="Screenshots/mnr-2-route-min.png" width=400 height=300"></td>
      <td valign="top"><<img src="Screenshots/sephla-route-min.png" width=400 height=300"></td>
    </tr>
</table>

##  Installation
To install the dependencies run: `pip install -r requirements.txt`. 

We also need to install the [Object Detection API for Tensorflow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md). Follow the instructions from the link to install the API.

The processed files for the data sets in the paper can be found in [Data Processed](https://github.com/dat181197/Big_Data_AQI_Estimation/tree/clean_codes/Data%20Processed) folder.

**Note**: *Emotion tags* and *Image features* are not available in the MediaEval 2019 data set.

The data are standardized and split into two sets.
The paper uses two object detection models *SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)* and *EfficientDet D7 1536x1536* for experiments which can be downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

## Model Training

### Pre-defined Params

Use the command `python train_models_with_params.py` to train the ML models using a pre-defined set of hyper-parameters. Detail usage and examples can be found [here](https://github.com/dat181197/Big_Data_AQI_Estimation/blob/master/Scripts/scripts%20input%20examples/train_models_with_params_inputs.md).

```
python train_models_with_params.py --help

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PROCESSED_DIR, --data_processed_dir DATA_PROCESSED_DIR
                        Directory of the train and test csv data (including
                        random split folders)
  -f {Sensor,Sensor+PW,Tag+Sensor,Tag+Sensor+PW,Tag+Sensor+Image,Tag+Sensor+Image+PW}, --feature_type {Sensor,Sensor+PW,Tag+Sensor,Tag+Sensor+PW,Tag+Sensor+Image,Tag+Sensor+Image+PW}
                        Type of data split
  -mc {SVM,Random Forest,Catboost,XGBoost,LightGBM,All}, --model_choice {SVM,Random Forest,Catboost,XGBoost,LightGBM,All}
                        Model name to use, use 'All' to use all available
                        models
  -p {pm25,pm10,o3,co,so2,no2,aqi}, --pollutant_to_predict {pm25,pm10,o3,co,so2,no2,aqi}
                        Specify name of pollutant to predict
  -om {SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50),EfficientDet D7 1536x1536}, --object_model_name {SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50),EfficientDet D7 1536x1536}
                        Name of the object model used to extact Image features
  -ms MODEL_SAVE_PATH, --model_save_path MODEL_SAVE_PATH
                        Path to save the model
  -rs RESULTS_SAVE_PATH, --results_save_path RESULTS_SAVE_PATH
                        Path to save output result
```
Example script to train SVM model on MNR data set resampled at 30S time window:
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

### Randomized Search

Use the command `python train_models_random_search.py` to train the ML models using a randomized search method with 5-fold cross validation. Detail usage and examples can be found [here](https://github.com/dat181197/Big_Data_AQI_Estimation/blob/master/Scripts/scripts%20input%20examples/train_models_random_search_inputs.md)

```
python train_models_random_search.py -h
optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PROCESSED_DIR, --data_processed_dir DATA_PROCESSED_DIR
                        Directory of the train and test csv data (including
                        random split folders)
  -f {Sensor,Sensor+PW,Tag+Sensor,Tag+Sensor+PW,Tag+Sensor+Image,Tag+Sensor+Image+PW}, --feature_type {Sensor,Sensor+PW,Tag+Sensor,Tag+Sensor+PW,Tag+Sensor+Image,Tag+Sensor+Image+PW}
                        Type of data split
  -mc {SVM,Random Forest,Catboost,XGBoost,LightGBM,All}, --model_choice {SVM,Random Forest,Catboost,XGBoost,LightGBM,All}
                        Model name to use, use 'All' to use all available
                        models
  -p {pm25,pm10,o3,co,so2,no2,aqi}, --pollutant_to_predict {pm25,pm10,o3,co,so2,no2,aqi}
                        Specify name of pollutant to predict
  -om {SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50),EfficientDet D7 1536x1536}, --object_model_name {SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50),EfficientDet D7 1536x1536}
                        Name of the object model used to extact Image features
  -ms MODEL_SAVE_PATH, --model_save_path MODEL_SAVE_PATH
                        Path to save the model
  -rs RESULTS_SAVE_PATH, --results_save_path RESULTS_SAVE_PATH
                        Path to save output result
```

Example script to train SVM model on MNR data set resampled at 30S time window:
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```
