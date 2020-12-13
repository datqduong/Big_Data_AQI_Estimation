# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:34:36 2020

@author: dat18
"""
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from combineFeatures import combinePublicWeather_
import argparse
from processData_MediaEval_2019 import combine_sensor_data_to_df, process_sensor_data
from utils import resample_data


def parse_arguments():
    ap = argparse.ArgumentParser()
    arg = ap.add_argument
        
    arg("--dataset_dir", type=str, required=True, help="Directory of data set")
    arg('--data_processed_dir', type=str, required=True, help="Directory of processed data and labels")
    
    arg('--feature_type', type=str, required=True, help="Type of data split", choices=["Sensor", "Sensor+PW"])#, "Tag+Sensor", "Tag+Sensor+PW", "Tag+Sensor+Image", "Tag+Sensor+Image+PW"])
    
    # arg('--object_model_name', type=str, help="Name of the object model used to extact Image features", choices=["SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)", "EfficientDet D7 1536x1536"])
        
    args = ap.parse_args()
    
    # if "Image" in args.feature_type and args.object_model_name is None:
    #     ap.error("--object_model_name argument is required when using this feature type")
    assert os.path.exists(os.path.realpath(os.path.expanduser(args.data_processed_dir))),\
            f"The directory '{args.data_processed_dir}' doesn't exist!"
        
    return args

def main(args):
    print("Args input entered: {}".format(args))

    data_path = args.dataset_dir
    data_processed_dir = args.data_processed_dir
    
    
    feature_type = args.feature_type
    # object_model_name = args.object_model_name
    
    # if object_model_name == "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)":
    #     object_model_folder = "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"
    # elif object_model_name == "EfficientDet D7 1536x1536":
    #     object_model_folder = "efficientdet_d7_coco17_tpu-32"
        
            
    if feature_type == "Sensor":
        data_folder_name = "Sensor Features"
        train_data_name = "train_data_sensor.csv"
        train_label_name = "train_label_sensor.csv"
        test_data_name = "test_data_sensor.csv"
        test_label_name = "test_label_sensor.csv"
        train_data_standardized_name = "train_data_sensor_standardized.csv"
        test_data_standardized_name = "test_data_sensor_standardized.csv"
        
        sensor_path = os.path.join(data_path, "Sensor Data")        
                             
        df = combine_sensor_data_to_df(sensor_path)
        
        print(f"MedEval19_sensor_data train size {df.shape}")
        
        # Process the whole sensor train data
        # _, sensor_data_processed, labels_data = process_sensor_data(df)
        
        # Resample sensor data based on a time-window
        sensor_data_resampled = resample_data(df, "30S")
        _, sensor_data_processed, labels_data = process_sensor_data(sensor_data_resampled)        
        
        X = sensor_data_processed.drop(columns=['timestamp']).apply(pd.to_numeric, errors='raise')
                  
    elif feature_type == "Sensor+PW":
        data_folder_name = "Sensor+PW Features"
        train_data_name = "train_data_sensor_pw.csv"
        train_label_name = "train_label_sensor_pw.csv"
        test_data_name = "test_data_sensor_pw.csv"
        test_label_name = "test_label_sensor_pw.csv"
        train_data_standardized_name = "train_data_sensor_pw_standardized.csv"
        test_data_standardized_name = "test_data_sensor_pw_standardized.csv"
        
        sensor_path = os.path.join(data_path, "Sensor Data")        
                             
        df = combine_sensor_data_to_df(sensor_path)
        
        print(f"MedEval19_sensor_data train size {df.shape}")
        
        # Process the whole sensor train data
        # _, sensor_data_processed, labels_data = process_sensor_data(df)
        
        # Resample sensor data based on a time-window
        sensor_data_resampled = resample_data(df, "30S")
        _, sensor_data_processed, labels_data = process_sensor_data(sensor_data_resampled)
                
        data_save_dir = os.path.join(data_processed_dir, data_folder_name)
        os.makedirs(data_save_dir, exist_ok=True)
        data_save_path = os.path.join(data_save_dir, "data+pw.csv")
        
        sensor_pw = combinePublicWeather_(sensor_data_processed, data_save_path)     
        
        X = sensor_pw.drop(columns=['timestamp']).apply(pd.to_numeric, errors='raise')
            
        
    random_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Random split')
    os.makedirs(random_split_dir, exist_ok=True)

    y = labels_data.apply(pd.to_numeric, errors='raise')
    
    print("[*] Checking null values:")
    print(X.isnull().sum())
    print(y.isnull().sum())

    print("[*] Splitting data and save to csv files")
    #%% Random split
    random_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Random split')
    os.makedirs(random_split_dir, exist_ok=True)
    #Split data to train, test sets with ratio .8:.2
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size= 0.2, random_state=24)
    
    X_train_random.to_csv(os.path.join(random_split_dir, train_data_name), index=False, sep=',', header=True)
    y_train_random.to_csv(os.path.join(random_split_dir, train_label_name), index=False, sep=',', header=True)
    
    X_test_random.to_csv(os.path.join(random_split_dir, test_data_name), index=False, sep=',', header=True)
    y_test_random.to_csv(os.path.join(random_split_dir, test_label_name), index=False, sep=',', header=True)
    
    #%% Standardize data
    scaler_random_split = preprocessing.StandardScaler().fit(X_train_random)
    
    X_train_random_standardized = pd.DataFrame(scaler_random_split.transform(X_train_random), columns=X_train_random.columns)
    X_test_random_standardized = pd.DataFrame(scaler_random_split.transform(X_test_random), columns=X_test_random.columns)
    
    X_train_random_standardized.to_csv(os.path.join(random_split_dir, train_data_standardized_name), index=False, sep=',', header=True)
    X_test_random_standardized.to_csv(os.path.join(random_split_dir, test_data_standardized_name), index=False, sep=',', header=True)
    
    print("[*] Train data, labels shape: {}, {}".format(X_train_random.shape, y_train_random.shape))
    print("[*] Test data, labels shape: {}, {}".format(X_test_random.shape, y_test_random.shape))
    print("[*] Done")
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)