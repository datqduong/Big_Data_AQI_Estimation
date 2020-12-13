# Multi-source Machine Learning for AQI Estimation
This repository contains the source code for the IEEE Big Data 2020 Conference: 6th Special Session on Intelligent Data Mining paper [Multi-source Machine Learning for AQI Estimation]() by Dat Q. Duong, Quang M. Le, Tan-Loc Nguyen-Tai, Dong Bo, Dat Nguyen, Minh-Son Dao, and Binh T. Nguyen.

##  Installation
To install the dependencies run: 
```
pip install -r requirements.txt
```

## Feature Engineering
To extract the feature information described in the paper, use the following scripts,
`processData_MNR.py`, `processData_MNR_Air.py`, `processData_MediaEval_2019.py`.
**Note**: *Emotion tags* and *Image features* are not available to process in the MediaEval 2019 data set.

Usage:
```
python processData_MNR.py --help

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        Directory of data set
  -s SAVE_DIR, --save_dir SAVE_DIR
                        Directory to save the processed data
  -om OBJECT_MODEL_PATH, --object_model_path OBJECT_MODEL_PATH
                        Directory of the model for object detection
  -tw {30S,60S}, --time_window {30S,60S}
                        Time window to re-sample sensor data
```
Example:
```
python processData_MNR.py -d "../../Data/Data HCM/_MNR_SENSOR_DATA_/sensor/" -s "../Data Processed/MNR Processed/ -om "../Object_Detection_Models/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8/" -tw "30S"
```

The processed csv files will be generated in the `-s` argument.
The argument `object_model_path` is the the path for the Object detection model, which can be downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

## Split data
To split the data set into train and hold out test data, use the `split_data.py` and `split_data_medeval19.py` scripts.

Usage:
```
python split_data.py --help

optional arguments:
  -h, --help            show this help message and exit
  --dataset_type {MNR,MNR_AIR}
                        Choose type of dataset using
  --data_processed_dir DATA_PROCESSED_DIR
                        Directory of processed data and labels
  --feature_type {Sensor,Sensor+PW,Tag+Sensor,Tag+Sensor+PW,Tag+Sensor+Image,Tag+Sensor+Image+PW}
                        Type of data split
  --object_model_name {SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50),EfficientDet D7 1536x1536}
                        Name of the object model used to extact Image features
```

The data will be standardized and split into two sets and saved at *--data_processed_dir*.
The paper used two object detection models *SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)* and *EfficientDet D7 1536x1536* for experiments. You can change the code to use other models for your needs. More example scripts can be found [here](https://github.com/dat181197/Big_Data_AQI_Estimation/blob/master/Scripts/scripts%20input%20examples/split_data%20inputs.md).

Example usage:
```
python split_data.py --dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor"
```
## Model Training

Use the command `python train_models_with_params.py` to train the ML models using a pre-defined set of hyper-parameters. Detail usage and examples can be found [here](https://github.com/dat181197/Big_Data_AQI_Estimation/blob/master/Scripts/scripts%20input%20examples/train_models_with_params_inputs.md).

Example script to train SVM model on MNR data set resampled at 30S time window:
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

Use the command `python train_models_random_search.py` to train the ML models using a randomized search method with 5-fold cross validation. Detail usage and examples can be found [here](https://github.com/dat181197/Big_Data_AQI_Estimation/blob/master/Scripts/scripts%20input%20examples/train_models_random_search_inputs.md)

```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```
