# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:29:27 2020

@author: dat18
"""
#%%
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from combineFeatures import combinePublicWeather
import argparse

ap = argparse.ArgumentParser()
arg = ap.add_argument

# arg('--dataset_type', type=str, required=True, help="Choose type of dataset using", choices=["ICMR", "ICMR_60S", "DDDA", "Hien", "Hien_15S", "Hien_30S", "Hien_60S", "Accumulated"])

arg('--dataset_type', type=str, required=True, help="Choose type of dataset using", choices = ["MNR", "MNR_AIR"])

# arg('--dataset_dir', type=str, required=True, help="Directory of the dataset")

arg('--data_processed_dir', type=str, required=True, help="Directory of processed data and labels")

arg('--feature_type', type=str, required=True, help="Type of data split", choices=["Sensor", "Sensor+PW", "Tag+Sensor", "Tag+Sensor+PW", "Tag+Sensor+Image", "Tag+Sensor+Image+PW"])

arg('--object_model_name', type=str, help="Name of the object model used to extact Image features", choices=["SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)", "EfficientDet D7 1536x1536"])
    
# arg('--feature_type', type=str, required=True, help="Type of data split", choices=["Sensor", "Sensor_Raw", "Images", "Combined", "Combined+GlobalWeather", "Sensor+GlobalWeather"])

# arg('--from_date', type=str, required=True, help="Specify the starting date to combine ('dd/mm/yyyy').")
# arg('--to_date', type=str, required=True, help="Specify the end date to combine ('dd/mm/yyyy').")

args = ap.parse_args()

def main():
    print("Args input entered: {}".format(args))
    
    if "Image" in args.feature_type and args.object_model_name is None:
        ap.error("--object_model_name argument is required when using this feature type")
        
    dataset_type = args.dataset_type
    # dataset_dir = args.dataset_dir
    # data_processed_dir = '../Data Processed/MNR Processed'
    # data_processed_dir = '../Data Processed/MNR Air Processed'
    data_processed_dir = args.data_processed_dir
    
    if not os.path.exists(data_processed_dir):
        raise Exception("Directory {} not exists, please check again".format(data_processed_dir))
        
    feature_type = args.feature_type
    object_model_name = args.object_model_name
    
    if object_model_name == "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)":
        object_model_folder = "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"
    elif object_model_name == "EfficientDet D7 1536x1536":
        object_model_folder = "efficientdet_d7_coco17_tpu-32"

    
    ###################################################################################
    if feature_type == "Sensor":
        data_folder_name = "Sensor Features"
        train_data_name = "train_data_sensor.csv"
        train_label_name = "train_label_sensor.csv"
        test_data_name = "test_data_sensor.csv"
        test_label_name = "test_label_sensor.csv"
        train_data_standardized_name = "train_data_sensor_standardized.csv"
        test_data_standardized_name = "test_data_sensor_standardized.csv"
        
        sensor_path = os.path.join(data_processed_dir, "30S Resampled")
        
        data_processed_path = os.path.join(sensor_path, dataset_type + "_sensor_data_resampled_processed.csv")
        labels_path = os.path.join(sensor_path, dataset_type + "_sensor_data_resampled_labels.csv")
        
        
        sensor_data_processed = pd.read_csv(data_processed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
           
        X = sensor_data_processed.drop(columns=["timestamp"]).apply(pd.to_numeric, errors='raise')
        
        
    elif feature_type == "Sensor+PW":
        data_folder_name = "Sensor+PW Features"
        train_data_name = "train_data_sensor_pw.csv"
        train_label_name = "train_label_sensor_pw.csv"
        test_data_name = "test_data_sensor_pw.csv"
        test_label_name = "test_label_sensor_pw.csv"
        train_data_standardized_name = "train_data_sensor_pw_standardized.csv"
        test_data_standardized_name = "test_data_sensor_pw_standardized.csv"
        
        sensor_path = os.path.join(data_processed_dir, "30S Resampled")
        
        data_processed_path = os.path.join(sensor_path, dataset_type + "_sensor_data_resampled_processed.csv")
        labels_path = os.path.join(sensor_path, dataset_type + "_sensor_data_resampled_labels.csv")
        
        sensor_data_processed = pd.read_csv(data_processed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
        
        data_save_dir = os.path.join(data_processed_dir, data_folder_name)
        os.makedirs(data_save_dir, exist_ok=True)
        sensor_pw = combinePublicWeather(sensor_data_processed, data_save_dir)
        X = sensor_pw.drop(columns=['timestamp']).apply(pd.to_numeric, errors='raise')
        
        
    elif feature_type == "Tag+Sensor":
        data_folder_name = "Tag+Sensor"
        train_data_name = "train_data_tag_sensor.csv"
        train_label_name = "train_label_tag_sensor.csv"
        test_data_name = "test_data_tag_sensor.csv"
        test_label_name = "test_label_tag_sensor.csv"
        train_data_standardized_name = "train_data_tag_sensor_standardized.csv"
        test_data_standardized_name = "test_data_tag_sensor_standardized.csv"
        
        data_processed_path = os.path.join(data_processed_dir, "Merged_No_Image_Features", dataset_type + "_merged_data_no_image_processed.csv")
        labels_path = os.path.join(data_processed_dir, "Merged_No_Image_Features", dataset_type + "_merged_data_no_image_labels.csv")
        
        tag_sensor_data_processed = pd.read_csv(data_processed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
            
        X = tag_sensor_data_processed.drop(columns=['timestamp', 'image_folder_path', 'image']).apply(pd.to_numeric, errors='raise')
        
        
    elif feature_type == "Tag+Sensor+PW":
        data_folder_name = "Tag+Sensor+PW"
        train_data_name = "train_data_tag_sensor_pw.csv"
        train_label_name = "train_label_tag_sensor_pw.csv"
        test_data_name = "test_data_tag_sensor_pw.csv"
        test_label_name = "test_label_tag_sensor_pw.csv"
        train_data_standardized_name = "train_data_tag_sensor_pw_standardized.csv"
        test_data_standardized_name = "test_data_tag_sensor_pw_standardized.csv"
        
        data_processed_path = os.path.join(data_processed_dir, "Merged_No_Image_Features", dataset_type + "_merged_data_no_image_processed.csv")
        labels_path = os.path.join(data_processed_dir, "Merged_No_Image_Features", dataset_type + "_merged_data_no_image_labels.csv")
       
        tag_sensor_data_processed = pd.read_csv(data_processed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
        
        data_save_dir = os.path.join(data_processed_dir, data_folder_name)
        os.makedirs(data_save_dir, exist_ok=True)
        tag_sensor_pw = combinePublicWeather(tag_sensor_data_processed, data_save_dir)
        X = tag_sensor_pw.drop(columns=['timestamp', 'image_folder_path', 'image']).apply(pd.to_numeric, errors='raise')

        
    elif feature_type == "Tag+Sensor+Image":
        data_folder_name = "Tag+Sensor+Image" + "_" + object_model_name
        train_data_name = "train_data_tag_sensor_image.csv"
        train_label_name = "train_label_tag_sensor_image.csv"
        test_data_name = "test_data_tag_sensor_image.csv"
        test_label_name = "test_label_tag_sensor_image.csv"
        train_data_standardized_name = "train_data_tag_sensor_image_standardized.csv"
        test_data_standardized_name = "test_data_tag_sensor_image_standardized.csv"
        
        data_procesed_path = os.path.join(data_processed_dir, "Merged_With_Image_Features", object_model_folder, dataset_type + "_merged_data_processed.csv")
        labels_path = os.path.join(data_processed_dir, "Merged_With_Image_Features", object_model_folder, dataset_type + "_merged_data_labels.csv")
        
        tag_sensor_image_data_processed = pd.read_csv(data_procesed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
        
        X = tag_sensor_image_data_processed.drop(columns=['timestamp', 'image_folder_path', 'image']).apply(pd.to_numeric, errors='raise')
        
    
    elif feature_type == "Tag+Sensor+Image+PW":
        data_folder_name = "Tag+Sensor+Image+PW" + "_" + object_model_name
        train_data_name = "train_data_tag_sensor_image_pw.csv"
        train_label_name = "train_label_tag_sensor_image_pw.csv"
        test_data_name = "test_data_tag_sensor_image_pw.csv"
        test_label_name = "test_label_tag_sensor_image_pw.csv"
        train_data_standardized_name = "train_data_tag_sensor_image_pw_standardized.csv"
        test_data_standardized_name = "test_data_tag_sensor_image_pw_standardized.csv"
        
        data_procesed_path = os.path.join(data_processed_dir, "Merged_With_Image_Features", object_model_folder, dataset_type + "_merged_data_processed.csv")
        labels_path = os.path.join(data_processed_dir, "Merged_With_Image_Features", object_model_folder, dataset_type + "_merged_data_labels.csv")
        
        tag_sensor_image_data_processed = pd.read_csv(data_procesed_path, parse_dates=["timestamp"])
        labels_data = pd.read_csv(labels_path)
        
        data_save_dir = os.path.join(data_processed_dir, data_folder_name)
        os.makedirs(data_save_dir, exist_ok=True)
        tag_sensor_image_pw = combinePublicWeather(tag_sensor_image_data_processed, data_save_dir)
        X = tag_sensor_image_pw.drop(columns=['timestamp', 'image_folder_path', 'image']).apply(pd.to_numeric, errors='raise')
        
    #%% Split data and save as csv
    
    # X = data_combined_features.drop(columns=['Time']).apply(pd.to_numeric, errors='raise', downcast = 'float')
    # y = labels_data.pm25.apply(pd.to_numeric, errors='raise', downcast ='float')
    # y = pd.DataFrame(columns=['pm25', 'aqi', 'aqi_rank'], data=labels_data[['pm25', 'aqi', 'aqi_rank']].values)
    # if feature_type == "Images":
    #     y = labels_data_image
    # else:
    #     y = labels_data
    y = labels_data.apply(pd.to_numeric, errors='raise')
    
    # y = y.apply(pd.to_numeric, errors='raise')
    
    # Check if there is still any categorical features
    print("Data types: \n", X.dtypes)
    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    # Check if there is null values
    print("Null values:")
    print(X.isnull().sum())
    print(y.isnull().sum())
    
    print("Splitting data...random")
    #%% Random split
    # random_split_dir = os.path.join(data_processed_dir, 'Combined Features + Global Weather', 'Random split')
    random_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Random split')
    os.makedirs(random_split_dir, exist_ok=True)
    #Split data to train, test sets with ratio .8:.2
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size= 0.2, random_state=24)
    
    # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size= 0.25, random_state=24)
    
    X_train_random.to_csv(os.path.join(random_split_dir, train_data_name), index=False, sep=',', header=True)
    y_train_random.to_csv(os.path.join(random_split_dir, train_label_name), index=False, sep=',', header=True)
    
    # X_validation.to_csv(os.path.join(random_split_dir, 'validation_data.csv'), index=False, sep=',', header=True)
    # y_validation.to_csv(os.path.join(random_split_dir, 'validation_label.csv'), index=False, sep=',', header=True)
    
    X_test_random.to_csv(os.path.join(random_split_dir, test_data_name), index=False, sep=',', header=True)
    y_test_random.to_csv(os.path.join(random_split_dir, test_label_name), index=False, sep=',', header=True)
    
    #%% Standardize data
    scaler_random_split = preprocessing.StandardScaler().fit(X_train_random)
    
    X_train_random_standardized = pd.DataFrame(scaler_random_split.transform(X_train_random), columns=X_train_random.columns)
    # X_validation_standardized = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
    X_test_random_standardized = pd.DataFrame(scaler_random_split.transform(X_test_random), columns=X_test_random.columns)
    
    X_train_random_standardized.to_csv(os.path.join(random_split_dir, train_data_standardized_name), index=False, sep=',', header=True)
    # X_validation_standardized.to_csv(os.path.join(data_processed_dir, 'validation_data_standardized.csv'), index=False, sep=',', header=True)
    X_test_random_standardized.to_csv(os.path.join(random_split_dir, test_data_standardized_name), index=False, sep=',', header=True)
    
    print("Train data, labels shape: {}, {}".format(X_train_random.shape, y_train_random.shape))
    print("Test data, labels shape: {}, {}".format(X_test_random.shape, y_test_random.shape))
    print("Done!")
    #%% Historical split, train from Feb-28 to Mar-05, test Mar-06
    # print("Splitting data...historical")
    # # hist_split_dir = os.path.join(data_processed_dir, 'Combined Features + Global Weather', 'Historical split')
    # hist_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Historical split')
    # os.makedirs(hist_split_dir, exist_ok=True)
    
    # if feature_type == "Images":
    #     day_06_index = image_features_with_labels.index[image_features_with_labels.day.isin([6])]
    #     other_days_index = ~image_features_with_labels.index.isin(day_06_index)
    # else:
    #     day_06_index = sensor_data_raw.index[sensor_data_raw.day.isin([6])]
    #     other_days_index = ~sensor_data_raw.index.isin(day_06_index)
                
    # # day_06_index = data_features.index[data_features['Timestamp'].dt.day.isin([6])]
    # X_test_hist = X.loc[day_06_index]
    # y_test_hist = y.loc[day_06_index]
    
    # # X_train_hist = X.loc[~data_features.index.isin(day_06_index)]
    # # y_train_hist = y.loc[~labels_data.index.isin(day_06_index)]
    # X_train_hist = X.loc[other_days_index]
    # y_train_hist = y.loc[other_days_index]
    
    # X_train_hist.to_csv(os.path.join(hist_split_dir, train_data_name), index=False, sep=',', header=True)
    # y_train_hist.to_csv(os.path.join(hist_split_dir, train_label_name), index=False, sep=',', header=True)
    
    # X_test_hist.to_csv(os.path.join(hist_split_dir, test_data_name), index=False, sep=',', header=True)
    # y_test_hist.to_csv(os.path.join(hist_split_dir, test_label_name), index=False, sep=',', header=True)
    
    # scaler_hist_split = preprocessing.StandardScaler().fit(X_train_hist)
    
    # X_train_hist_standardized = pd.DataFrame(scaler_hist_split.transform(X_train_hist), columns=X_train_hist.columns)
    # X_test_hist_standardized = pd.DataFrame(scaler_hist_split.transform(X_test_hist), columns=X_test_hist.columns)
    
    # X_train_hist_standardized.to_csv(os.path.join(hist_split_dir, train_data_standardized_name), index=False, sep=',', header=True)
    # X_test_hist_standardized.to_csv(os.path.join(hist_split_dir, test_data_standardized_name), index=False, sep=',', header=True)
    
    # print("Train data, labels shape: {}, {}".format(X_train_hist.shape, y_train_hist.shape))
    # print("Test data, labels shape: {}, {}".format(X_test_hist.shape, y_test_hist.shape))
    # print("Done!")

if __name__ == "__main__":
    main()