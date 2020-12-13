# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:19:15 2020

@author: dat18
"""
from features import Image_features
from features.TL_features import get_part_of_day, is_RushHour, distance_to_airport_
from utils import calc_AQI_from_df
import glob
import os
from utils import resample_data, create_empty_pollutants_columns_
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import random


def parse_arguments():
    ap = argparse.ArgumentParser()
    arg = ap.add_argument
    
    arg("-d", "--dataset_dir", type=str, required=True, help="Directory of data set")
    arg("-s", "--save_dir", type=str, required=True, help="Directory to save the processed data")
    arg("-om", "--object_model_path", type=str, required=True, help="Directory of the model for object detection")
    arg("-tw", "--time_window", type=str, required=True, help="Time window to re-sample sensor data", choices = ["30S", "60S"])
    
    args = ap.parse_args()
    
    assert os.path.exists(os.path.realpath(os.path.expanduser(args.dataset_dir))),\
    f"The directory '{args.dataset_dir}' doesn't exist!"
    
    return args


def combine_all_available_merged_data(data_path):
    print(f"[*] Reading all available merged data from {os.path.abspath(data_path)}")
    all_files = glob.glob(os.path.join(data_path, "**/*.csv", recursive=True))
    
    df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)
    
    return df
    
    
def combine_merged_data_to_df(data_path):
    df = pd.DataFrame()
    print(f"[*] Reading merged data from {os.path.abspath(data_path)}...")
    all_files = []
    random.seed(24)
    for idx, date_folder in enumerate(os.listdir(data_path)):
        date = datetime.strptime(date_folder, "%Y%m%d")
        users_file_list = glob.glob(os.path.join(data_path, date_folder, "*.csv"))
        file_name = random.choice(users_file_list)
        user_name = os.path.basename(file_name).split('-emotion-tag-' + date.strftime("%d-%m-%Y"))[0]
        photo_folder_name = "image-tag-" + user_name + "-" + date.strftime("%d-%m-%Y")
        photo_folder_full_path = os.path.relpath(os.path.join(data_path, date_folder, photo_folder_name))
        all_files.append([file_name, photo_folder_full_path])
    
    for f, photo_folder_path in all_files:
        file_df = pd.read_csv(f)
        image_links_column = file_df['Image Links']
        image_name_column = image_links_column.apply(lambda row: os.path.basename(row))
        file_df.insert(1, 'image_folder_path', photo_folder_path)
        file_df.insert(2, 'image name', image_name_column)
        file_df.drop(columns=["Image Links"], inplace=True)
        df = pd.concat([df, file_df], axis=0, ignore_index=True)
    
    df.columns = df.columns.str.lower()
    
    df['time'] = pd.to_datetime(df['time'])
    print("[*] Sorting values based on Timestamp...")
    df = df.sort_values('time')

    print("[*] Removing duplicates....")
    df.drop_duplicates('time', inplace=True, ignore_index=True)
    
    # df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    df.drop(columns=['uv', 'obstruction', 'what the most suitable vehicle for using the route?'], inplace=True) # Remove UV which has lots of errors
    
    print("[*] Check null values...")
    print(df.isnull().sum())
    
    print("[*] Removing null values...")
    df.dropna(axis=0, inplace=True)
    
    print("Done")
    return df


def combine_sensor_data_to_df(data_path):
    print(f"[*] Reading sensor data from {os.path.abspath(data_path)}")

    random.seed(24)
    
    all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    
    df_train = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index = True)
    df_train.columns = df_train.columns.str.lower()
    
    print("[*] Sorting values based on Timestamp...")

    df_train['time'] = pd.to_datetime(df_train['time'])
    df_train['time'] = df_train['time'].dt.tz_localize(None)
    
    df_train = df_train.sort_values('time')
    df_train.reset_index(inplace=True, drop=True)
    
    # Remove no2 and o3 which has lots of errors
    df_train.drop(columns=['no2', 'o3'], inplace=True) 
    
    # Remove 0 values in pm2.5
    df_train = df_train.iloc[df_train['pm25'].to_numpy().nonzero()]
    
    print("[*] Check null values in train set...")
    print(df_train.isnull().sum())
    
    print("[*] Removing null values in train set...")
    df_train.dropna(axis=0, inplace=True)
    
    print("[*] Removing duplicates values...")
    df_train.drop_duplicates(subset='time', keep='first', inplace=True, ignore_index=True)
    
    print("[*] Removing inconsistent values...")
    threshold = 200
    df_train = df_train[df_train['pm25'] < threshold]
    
    print("Done")
    return df_train

        
def get_timestamp_features(dataframe):
    print("[*] Extracting timestamp and location features...")
    # Features from timestamp include
    # Morning/Noon/Afternoon/Evening
    # is RushHour (7AM to 8:30AM and 4:30 PM to 6:30 PM) from Monday to Saturday
    result_df = dataframe.copy()
    result_df.drop(columns=['lat', 'lon'], inplace=True)
    datetime_column = dataframe['timestamp']
    # dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    # Split date, time
    day = datetime_column.dt.day
    month = datetime_column.dt.month
    year = datetime_column.dt.year
    time = datetime_column.dt.time
    hour = datetime_column.dt.hour
    
    dataframe.insert(1, "day", day)
    dataframe.insert(2, "month", month)
    dataframe.insert(3, "year", year)
    dataframe.insert(4, "time", time)
    
    # result_df.insert(0, "timestamp", datetime_column)
    
    # Add part_of_day and is_rush_hour features
    period = hour.apply(get_part_of_day)
    # Convert to one-hot encoding
    pod_df = pd.get_dummies(period, prefix="part_of_day", dtype=np.float64)
    result_df = pd.concat([result_df, pod_df], axis=1)
    dataframe.insert(5, "partofday", period)
    
    isRushHour = hour.apply(is_RushHour)
    isRH_df = pd.get_dummies(isRushHour, dtype=np.float64)
    result_df = pd.concat([result_df, isRH_df], axis=1)
    dataframe.insert(6, "is_RushHour", isRushHour)
    
    # Location features, "isNearAirport", "distanceToAirport"
    location = dataframe[['lat', 'lon']]
    
    distanceToAirport = location.apply(distance_to_airport_, axis = 1)
    result_df["distance_to_airport"] = distanceToAirport
    dataframe.insert(7, "distance_to_airport", distanceToAirport)
     
    return dataframe, result_df
                
def process_sensor_data(df):
    ################### Extracting the labels #########################
    
    # Get pollutants values
    # air_data_df = df[['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']].copy()
    air_data_df = df[['pm25']].copy()
    
    # air_data_df = air_data_df.reindex(columns=['o3_8', 'o3', 'pm10', 'pm25', 'co', 'so2', 'so2_24', 'no2'])
    empty_pol = ['o3_8', 'so2_24', 'pm10', 'co', 'so2', 'no2', 'o3']
    air_data_df = create_empty_pollutants_columns_(air_data_df, empty_pol, ['pm25'])
    
    air_data_np = air_data_df.to_numpy()
    rows, cols = air_data_np.shape
    
    # Calculate AQI column
    AQI_list_np, AQI_rank_np = calc_AQI_from_df(air_data_df)
    
    # Add AQI column to dataframe
    air_data_df['aqi'] = AQI_list_np
    air_data_df['aqi_rank'] = AQI_rank_np
    ###################################################################
    # Add timestamp features
    
    df_copy = df.rename(columns={'time': 'timestamp'})
        
    df_copy = df_copy.drop(columns=['pm25']) # remove unecessary columns
    
    result_raw, result_processed = get_timestamp_features(df_copy)
    
    return result_raw, result_processed, air_data_df

def process_merged_data(df, model_path):
    ################### Extracting the labels #########################
    
    # Get pollutants values
    air_data_df = df[['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']].copy()
    
    # Add zeros values to missing pollutants
    air_data_df.insert(0, 'o3_8', 0)
    air_data_df.insert(6, 'so2_24', 0)
    
    air_data_df = air_data_df.reindex(columns=['o3_8', 'o3', 'pm10', 'pm25', 'co', 'so2', 'so2_24', 'no2'])
    
    air_data_np = air_data_df.to_numpy()
    rows, cols = air_data_np.shape
    
    # Calculate AQI column
    AQI_list_np, AQI_rank_np = calc_AQI_from_df(air_data_df)
    
    # Add AQI column to dataframe
    air_data_df['aqi'] = AQI_list_np
    air_data_df['aqi_rank'] = AQI_rank_np
    ###################################################################
    ################# Process timestamp features ######################
    df_copy = df.rename(columns=
                        {'time': 'timestamp', \
                        'image name': 'image', \
                        'greenness degree: 1 (building) -> 5 (greenness)': 'greenness degree',\
                        'cleanliness degree: 1 (filthy) -> 5 (cleanliness)': 'cleanliness degree',\
                        'crowdedness degree: (vehicle density: 1 (very light) -> 5 (high dense)/pedestrian density: 1 (very light) -> 5 (high dense)': 'crowdedness degree',\
                        'noisy degree: 1 (very quiet) -> 5 (very noisy)': 'noisy degree',\
                        'skin feeling degree: 1 (bad) -> 5 (good)': 'skin feeling degree',\
                        'stress degree: ( 1stressed,  2depressed,  3calm,  4relaxed,  5excited)': 'stress degree',\
                        'personal aqi degree: 1 (fresh air) -> 5 (absolute pollution)': 'personal aqi degree',\
                        'do you want to use this route so that you can protect your health and safety (i.e., avoid air pollution, congestion, and obstruction)? safety degree: 1 (not want at all) -> 5': 'safety degree'
                        })
    
    df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2'], inplace=True) # remove unecessary columns
    
    df_copy.insert(4, 'lat', df_copy.location.str.split(",").str[0])
    df_copy.insert(5, 'lon', df_copy.location.str.split(",").str[1])
    
    df_copy['lat'] = pd.to_numeric(df_copy['lat'], errors='raise')
    df_copy['lon'] = pd.to_numeric(df_copy['lon'], errors='raise')
    
    df_copy.drop(columns=["location"], inplace=True)
    result_raw, result_processed = get_timestamp_features(df_copy)
    ####################################################################
    ################ Process image features ############################
    image_path = result_processed["image_folder_path"] + os.sep + result_processed["image"]
    image_path = image_path.to_frame("image_path")
    image_features = Image_features.get_image_features(image_path, model_path)
    
    result_raw = pd.concat([result_raw, image_features], axis=1)
    result_processed = pd.concat([result_processed, image_features], axis=1)
    
    return result_raw, result_processed, air_data_df

def process_merged_data_no_image(df):
    ################### Extracting the labels #########################
    
    # Get pollutants values
    air_data_df = df[['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']].copy()
    
    # Add zeros values to missing pollutants
    air_data_df.insert(0, 'o3_8', 0)
    air_data_df.insert(6, 'so2_24', 0)
    
    air_data_df = air_data_df.reindex(columns=['o3_8', 'o3', 'pm10', 'pm25', 'co', 'so2', 'so2_24', 'no2'])
    
    air_data_np = air_data_df.to_numpy()
    rows, cols = air_data_np.shape
    
    # Calculate AQI column
    AQI_list_np, AQI_rank_np = calc_AQI_from_df(air_data_df)
    
    # Add AQI column to dataframe
    air_data_df['aqi'] = AQI_list_np
    air_data_df['aqi_rank'] = AQI_rank_np
    ###################################################################
    ################# Process timestamp features ######################
    df_copy = df.rename(columns=
                        {'time': 'timestamp', \
                        'image name': 'image', \
                        'greenness degree: 1 (building) -> 5 (greenness)': 'greenness degree',\
                        'cleanliness degree: 1 (filthy) -> 5 (cleanliness)': 'cleanliness degree',\
                        'crowdedness degree: (vehicle density: 1 (very light) -> 5 (high dense)/pedestrian density: 1 (very light) -> 5 (high dense)': 'crowdedness degree',\
                        'noisy degree: 1 (very quiet) -> 5 (very noisy)': 'noisy degree',\
                        'skin feeling degree: 1 (bad) -> 5 (good)': 'skin feeling degree',\
                        'stress degree: ( 1stressed,  2depressed,  3calm,  4relaxed,  5excited)': 'stress degree',\
                        'personal aqi degree: 1 (fresh air) -> 5 (absolute pollution)': 'personal aqi degree',\
                        'do you want to use this route so that you can protect your health and safety (i.e., avoid air pollution, congestion, and obstruction)? safety degree: 1 (not want at all) -> 5': 'safety degree'
                        })
    
    df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2'], inplace=True) # remove unecessary columns
    
    df_copy.insert(4, 'lat', df_copy.location.str.split(",").str[0])
    df_copy.insert(5, 'lon', df_copy.location.str.split(",").str[1])
    
    df_copy['lat'] = pd.to_numeric(df_copy['lat'], errors='raise')
    df_copy['lon'] = pd.to_numeric(df_copy['lon'], errors='raise')
    
    df_copy.drop(columns=["location"], inplace=True)
    result_raw, result_processed = get_timestamp_features(df_copy)
    ####################################################################
    return result_raw, result_processed, air_data_df
    
def main():
    # Process arguments
    args = parse_arguments()
    # data_path = "../../Data/Data HCM/DDDA_Data"
    # model_path = '../../Object_Detection_Models/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8/'
    data_path = args.dataset_dir
    model_path = args.object_model_path
    save_dir = args.save_dir
    time_window = args.time_window
    
    model_path = os.path.join(model_path, "saved_model")
    
    # Create folder for full sensor data
    full_sensor_folder_name = "Full Sensor"
    full_sensor_folder_path = os.path.join(save_dir, full_sensor_folder_name)
    os.makedirs(full_sensor_folder_path, exist_ok=True)
    
    # Create folder for resample sensor data
    resample_save_folder_name = time_window + " Resampled"
    resample_save_folder_path = os.path.join(save_dir, resample_save_folder_name)
    os.makedirs(resample_save_folder_path, exist_ok=True)
    
    # We only need to process sensor only data, and merged data
    sensor_path = os.path.join(data_path, "Sensor Data")
    
    # Read sensor data to df
    MedEval19_sensor_data = combine_sensor_data_to_df(sensor_path)
    
    print(f"MedEval19_sensor_data full size {MedEval19_sensor_data.shape}")
    # Process the whole sensor data
    MedEval19_sensor_data_raw, MedEval19_sensor_data_processed, MedEval19_sensor_data_labels = process_sensor_data(MedEval19_sensor_data)
    
    print("[*] Checking nulls...")
    print(MedEval19_sensor_data_raw.isnull().sum())
    print(MedEval19_sensor_data_processed.isnull().sum())
    print(MedEval19_sensor_data_labels.isnull().sum())
    
    print(f"Resampled data size {MedEval19_sensor_data_raw.shape}")
    print(f"Resampled data processed size {MedEval19_sensor_data_processed.shape}")
    print(f"Resampled data label size {MedEval19_sensor_data_labels.shape}")

    # Save the processed data to csv file
    MedEval19_sensor_data_raw_file_name = os.path.join(full_sensor_folder_path, "MedEval19_sensor_data_raw.csv")
    MedEval19_sensor_data_resampled_processed_file_name = os.path.join(full_sensor_folder_path, "MedEval19_sensor_data_processed.csv")
    MedEval19_sensor_data_resampled_labels_file_name = os.path.join(full_sensor_folder_path, "MedEval19_sensor_data_labels.csv")
    
    
    print("[*] Saving resampled sensor data...")
    MedEval19_sensor_data_raw.to_csv(MedEval19_sensor_data_raw_file_name, index=False, sep=',', header=True)
    MedEval19_sensor_data_processed.to_csv(MedEval19_sensor_data_resampled_processed_file_name, index=False, sep=',', header=True)
    MedEval19_sensor_data_labels.to_csv(MedEval19_sensor_data_resampled_labels_file_name, index=False, sep=',', header=True)
    print("Done")


    # Resample sensor data based on a time-window
    MedEval19_sensor_data_resampled = resample_data(MedEval19_sensor_data, time_window)
    
    print("[*] Checking nulls...")
    print(MedEval19_sensor_data_resampled.isnull().sum())
    
    # Process the resampled sensor data
    print("[*] Processing resampled sensor data....")
    MedEval19_sensor_data_resampled_raw, MedEval19_sensor_data_resampled_processed, MedEval19_sensor_data_resampled_labels = process_sensor_data(MedEval19_sensor_data_resampled)
    
    print("[*] Checking nulls...")
    print(MedEval19_sensor_data_resampled_raw.isnull().sum())
    print(MedEval19_sensor_data_resampled_processed.isnull().sum())
    print(MedEval19_sensor_data_resampled_labels.isnull().sum())
    

    print(f"Resampled data size {MedEval19_sensor_data_resampled_raw.shape}")
    print(f"Resampled data processed size {MedEval19_sensor_data_resampled_processed.shape}")
    print(f"Resampled data label size {MedEval19_sensor_data_resampled_labels.shape}")
    
    # Save the processed data to csv file
    MedEval19_sensor_data_resampled_raw_file_name = os.path.join(resample_save_folder_path, "MedEval19_sensor_data_resampled_raw.csv")
    MedEval19_sensor_data_resampled_processed_file_name = os.path.join(resample_save_folder_path, "MedEval19_sensor_data_resampled_processed.csv")
    MedEval19_sensor_data_resampled_labels_file_name = os.path.join(resample_save_folder_path, "MedEval19_sensor_data_resampled_labels.csv")
    
    
    print("[*] Saving resampled sensor data...")
    MedEval19_sensor_data_resampled_raw.to_csv(MedEval19_sensor_data_resampled_raw_file_name, index=False, sep=',', header=True)
    MedEval19_sensor_data_resampled_processed.to_csv(MedEval19_sensor_data_resampled_processed_file_name, index=False, sep=',', header=True)
    MedEval19_sensor_data_resampled_labels.to_csv(MedEval19_sensor_data_resampled_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
if __name__ == "__main__":
    main()
    
    
    
    