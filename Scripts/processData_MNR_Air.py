# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 20:35:05 2020

@author: dat18
"""
from features import Image_features
from features.TL_features import get_part_of_day, is_RushHour, distance_to_airport
from utils import calc_AQI_from_df
import glob
import os
from utils import resample_data
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import random


user_map = {"user1": "dat", "user2": "dong", "user3": "duy-anh"}

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

    # user_pick = pick_user_per_day()
    # Combine all available data into DataFrame
    
    all_files = []
    random.seed(24)
    # Get all .csv files in sensor folder recursively
    for idx, date_folder in enumerate(os.listdir(data_path)):
        sensor_folder = os.path.join(data_path, date_folder, "sensor_data")
        # user_choice = user_pick[idx]
        user_folders = os.listdir(sensor_folder)
        user_choice = random.choice(user_folders)
        file_of_day = glob.glob(os.path.join(sensor_folder, user_choice, '*.csv'))
        
        all_files.append(file_of_day[0])
    # all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    
    df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index = True)
    df.columns = df.columns.str.lower()
    print("[*] Sorting values based on Timestamp...")

    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df = df.sort_values('time')
    df.reset_index(inplace=True, drop=True)
    
    df.drop(columns=['uv', 'image'], inplace=True) # Remove UV which has lots of errors
    
    print("[*] Check null values...")
    print(df.isnull().sum())
    
    print("[*] Removing null values...")
    df.dropna(axis=0, inplace=True)
    
    print("Done")
    return df
        
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
    distanceToAirport = location.apply(distance_to_airport, axis = 1)
    result_df["distance_to_airport"] = distanceToAirport
    dataframe.insert(7, "distance_to_airport", distanceToAirport)
     
    # Combine result with weather data in dataframe
    # weather_data = dataframe[['hum', 'tem', 'uv']]
    # result_df = pd.concat([result_df, weather_data], axis=1)
    return dataframe, result_df
                
def process_sensor_data(df):
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
    # Add timestamp features
    
    df_copy = df.rename(columns={'time': 'timestamp'})
        
    df_copy = df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2']) # remove unecessary columns
    
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
    
    # Create folder for resample sensor data
    resample_save_folder_name = time_window + " Resampled"
    resample_save_folder_path = os.path.join(save_dir, resample_save_folder_name)
    os.makedirs(resample_save_folder_path, exist_ok=True)
    
    # Create folder for merged data no image
    merged_no_image_save_folder_name = "Merged_No_Image_Features"
    merged_no_image_save_folder_path = os.path.join(save_dir, merged_no_image_save_folder_name)
    os.makedirs(merged_no_image_save_folder_path, exist_ok=True)
    
    # Create folder for merged data with image folder for each object detection model used
    merged_with_image_save_folder_name = "Merged_With_Image_Features"
    model_path_norm = os.path.normpath(model_path)
    object_model_name = model_path_norm.split(os.sep)[-2]
    merged_save_folder_path = os.path.join(save_dir, merged_with_image_save_folder_name, object_model_name)
    os.makedirs(merged_save_folder_path, exist_ok=True)
    
    # For this data we already have merged data that's been merged from sensor to emotion_tag

    # We only need to process sensor only data, and merged data
    sensor_path = os.path.join(data_path, "mapping_image_into_sensor_update")
    
    # Read sensor data to df
    MNR_AIR_sensor_data = combine_sensor_data_to_df(sensor_path)
    
    print(f"MNR_AIR_sensor_data full size {MNR_AIR_sensor_data.shape}")
    # Process the whole sensor data
    # MNR_AIR_sensor_data_raw, MNR_AIR_sensor_data_processed, MNR_AIR_whole_data_labels = process_sensor_data(MNR_AIR_sensor_data)

    # Resample sensor data based on a time-window
    MNR_AIR_sensor_data_resampled = resample_data(MNR_AIR_sensor_data, time_window)
    
    print("[*] Checking nulls...")
    print(MNR_AIR_sensor_data_resampled.isnull().sum())
    
    # Process the resampled sensor data
    print("[*] Processing resampled sensor data....")
    MNR_AIR_sensor_data_resampled_raw, MNR_AIR_sensor_data_resampled_processed, MNR_AIR_sensor_data_resampled_labels = process_sensor_data(MNR_AIR_sensor_data_resampled)
    
    print("[*] Checking nulls...")
    print(MNR_AIR_sensor_data_resampled_raw.isnull().sum())
    print(MNR_AIR_sensor_data_resampled_processed.isnull().sum())
    print(MNR_AIR_sensor_data_resampled_labels.isnull().sum())
    

    print(f"Resampled data size {MNR_AIR_sensor_data_resampled_raw.shape}")
    print(f"Resampled data processed size {MNR_AIR_sensor_data_resampled_processed.shape}")
    print(f"Resampled data label size {MNR_AIR_sensor_data_resampled_labels.shape}")
    
    # Save the processed data to csv file
    MNR_AIR_sensor_data_resampled_raw_file_name = os.path.join(resample_save_folder_path, "MNR_AIR_sensor_data_resampled_raw.csv")
    MNR_AIR_sensor_data_resampled_processed_file_name = os.path.join(resample_save_folder_path, "MNR_AIR_sensor_data_resampled_processed.csv")
    MNR_AIR_sensor_data_resampled_labels_file_name = os.path.join(resample_save_folder_path, "MNR_AIR_sensor_data_resampled_labels.csv")
    
    
    print("[*] Saving resampled sensor data...")
    MNR_AIR_sensor_data_resampled_raw.to_csv(MNR_AIR_sensor_data_resampled_raw_file_name, index=False, sep=',', header=True)
    MNR_AIR_sensor_data_resampled_processed.to_csv(MNR_AIR_sensor_data_resampled_processed_file_name, index=False, sep=',', header=True)
    MNR_AIR_sensor_data_resampled_labels.to_csv(MNR_AIR_sensor_data_resampled_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
    
    # Read the merged data
    merged_data_path = os.path.join(data_path, "merged_data")
    MNR_AIR_merged_data = combine_merged_data_to_df(merged_data_path)
    
    # Process the merged data without image features
    print("[*] Processing merged data without image features...")
    MNR_AIR_merged_data_no_image_raw, MNR_AIR_merged_data_no_image_processed, MNR_AIR_merged_data_no_image_labels = process_merged_data_no_image(MNR_AIR_merged_data)
    print("Done")
    
    print("[*] Checking nulls...")
    print(MNR_AIR_merged_data_no_image_raw.isnull().sum())
    print(MNR_AIR_merged_data_no_image_processed.isnull().sum())
    print(MNR_AIR_merged_data_no_image_labels.isnull().sum())
    
    print(f"Merged data without image features size {MNR_AIR_merged_data_no_image_raw.shape}")
    print(f"Merged data without image features processed size {MNR_AIR_merged_data_no_image_processed.shape}")
    print(f"Merged labels without image size {MNR_AIR_merged_data_no_image_labels.shape}")
    
    # Save the processed data to csv file
    MNR_AIR_merged_data_no_image_raw_file_name = os.path.join(merged_no_image_save_folder_path, "MNR_AIR_merged_data_no_image_raw.csv")
    MNR_AIR_merged_data_no_image_processed_file_name = os.path.join(merged_no_image_save_folder_path, "MNR_AIR_merged_data_no_image_processed.csv")
    MNR_AIR_merged_data_no_image_labels_file_name = os.path.join(merged_no_image_save_folder_path, "MNR_AIR_merged_data_no_image_labels.csv")
    print("[*] Saving merged data without image features....")
    MNR_AIR_merged_data_no_image_raw.to_csv(MNR_AIR_merged_data_no_image_raw_file_name, index=False, sep=',', header=True)
    MNR_AIR_merged_data_no_image_processed.to_csv(MNR_AIR_merged_data_no_image_processed_file_name, index=False, sep=',', header=True)
    MNR_AIR_merged_data_no_image_labels.to_csv(MNR_AIR_merged_data_no_image_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
    
    # Process the merged data with image features
    print("[*] Processing merged data with image features...")
    MNR_AIR_merged_data_raw, MNR_AIR_merged_data_processed, MNR_AIR_merged_data_labels = process_merged_data(MNR_AIR_merged_data, model_path)
    print("Done")
    
    print("[*] Checking nulls...")
    print(MNR_AIR_merged_data_raw.isnull().sum())
    print(MNR_AIR_merged_data_processed.isnull().sum())
    print(MNR_AIR_merged_data_labels.isnull().sum())
    
    print(f"Merged data with image features size {MNR_AIR_merged_data_raw.shape}")
    print(f"Merged data  with image features processed size {MNR_AIR_merged_data_processed.shape}")
    print(f"Merged data with image features labels size {MNR_AIR_merged_data_labels.shape}")
    
    MNR_AIR_merged_data_raw_file_name = os.path.join(merged_save_folder_path, "MNR_AIR_merged_data_raw.csv")
    MNR_AIR_merged_data_processed_file_name = os.path.join(merged_save_folder_path, "MNR_AIR_merged_data_processed.csv")
    MNR_AIR_merged_data_labels_file_name = os.path.join(merged_save_folder_path, "MNR_AIR_merged_data_labels.csv")
    
    # Save to csv files
    print("[*] Saving merged data with image features...")
    MNR_AIR_merged_data_raw.to_csv(MNR_AIR_merged_data_raw_file_name, index=False, sep=',', header=True)
    MNR_AIR_merged_data_processed.to_csv(MNR_AIR_merged_data_processed_file_name, index=False, sep=',', header=True)
    MNR_AIR_merged_data_labels.to_csv(MNR_AIR_merged_data_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
    
if __name__ == "__main__":
    main()
    
    