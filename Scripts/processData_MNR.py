# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:55:55 2020

@author: dat18
"""
from features import Image_features
from features.TL_features import get_part_of_day, is_RushHour, distance_to_airport
from utils import calc_AQI_from_df
import glob
import os
from utils import resample_data
import argparse

import numpy as np
import pandas as pd

def parse_arguments():
    ap = argparse.ArgumentParser()
    arg = ap.add_argument
    
    arg("--dataset_dir", type=str, required=True, help="Directory of data set")
    arg("--save_dir", type=str, required=True, help="Directory to save the processed data")
    arg("--object_model_path", type=str, required=True, help="Directory of the model for object detection")
    arg("--time_window", type=str, required=True, help="Time window to re-sample sensor data", choices = ["30S", "60S"])
    
    args = ap.parse_args()
    return args

# def combine_mapped_data_to_df(data_path):
#     print(f"[*] Reading mapped data from {os.path.abspath(data_path)}...")
#     all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    
#     df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index = True)
#     df.columns = df.columns.str.upper()
    
#     df['TIME'] = pd.to_datetime(df['TIME'])
#     print("[*] Sorting values based on Timestamp...")
#     df = df.sort_values('TIME')
#     # df.dropna(axis=1, inplace=True)
#     df.reset_index(inplace=True, drop=True)
#     print("Done")
#     return df

def combine_emotion_tags_to_df(data_path):
    print(f"[*] Reading emtion tags from {os.path.abspath(data_path)}")
    # Combine all available data into DataFrame

    # Get all .csv files in sensor folder recursively
    
    # Faster, Python 3.5+ only
    all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    
    df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index = True)
    
    print("[*] Removing null values...")
    df.drop(columns={"Users", "Obstruction", "Congestion", "Traffic_Light", "Pothole", "Construction_Site", "Vehicle for using the route"}, inplace=True)
    df.dropna(axis=0, inplace=True)
    
    photo_folder_full_path = os.path.relpath(os.path.join(data_path, 'image_tag'))
    df.insert(1, 'image_folder_path', photo_folder_full_path)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], dayfirst=True)
    df.rename(columns={"Date_Time": "time"}, inplace=True)
    df.columns = df.columns.str.lower()
    
    print("[*] Sorting values based on Timestamp...")
    df = df.sort_values('time')
    
    print("[*] Removing duplicates...")
    df.drop_duplicates('time', inplace=True, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    print("Done")
    return df

def combine_sensor_data_to_df(data_path):
    print(f"[*] Reading sensor data from {os.path.abspath(data_path)}")

    # Combine all available data into DataFrame
    
    # Get all .csv files in sensor folder recursively
    all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    
    df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index = True)
    print("[*] Sorting values based on Timestamp...")

    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df = df.sort_values('time')
    df.reset_index(inplace=True, drop=True)
    print("Done")
    return df

def merge_sensor_to_tag(sensor_data, emtion_tag_data, on="time"):
    print("[*] Merging sensor data to emotion tags...")
    sensor_data_drop = sensor_data.drop(columns=['lat', 'lon', 'fah', 'image', 'heartbeat'])
    merged_df = pd.merge_asof(emtion_tag_data, sensor_data_drop, on=on)
    merged_df.dropna(axis=0, inplace=True)
    merged_df.reset_index(inplace=True, drop=True)
    print("Done")
    return merged_df
        
def get_timestamp_features(dataframe):
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
    # isNearAirport = location.apply(is_NearAirPort, axis=1)
    # isNearAP_df = pd.get_dummies(isNearAirport, dtype=np.float64)
    # result_df = pd.concat([result_df, isNearAP_df], axis=1)
    # dataframe.insert(3, "is_NearAirPort", isNearAirport)
    
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
        
    df_copy = df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2', 'heartbeat', 'fah']) # remove unecessary columns
    
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
                        {'time': 'timestamp',
                         'greenness_degree': 'greenness degree',
                         'cleanliness_degree': 'cleanliness degree',
                         'crowdedness_degree': 'crowdedness degree',
                         'noisy_degree': 'noisy degree',
                         'skin_feeling_degree': 'skin feeling degree',
                         'stress_degree': 'stress degree',
                         'personal_aqi_degree': 'personal aqi degree',
                         ' health_and_safety': 'safety degree'                        
                        })
    df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2'], inplace=True)
    
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
                        {'time': 'timestamp',
                         'greenness_degree': 'greenness degree',
                         'cleanliness_degree': 'cleanliness degree',
                         'crowdedness_degree': 'crowdedness degree',
                         'noisy_degree': 'noisy degree',
                         'skin_feeling_degree': 'skin feeling degree',
                         'stress_degree': 'stress degree',
                         'personal_aqi_degree': 'personal aqi degree',
                         ' health_and_safety': 'safety degree'                        
                        })
    df_copy.drop(columns=['o3', 'pm10', 'pm25', 'co', 'so2', 'no2'], inplace=True)
    
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
    
    # data_path = "../../Data/Data HCM/_MNR_SENSOR_DATA_"
    data_path = args.dataset_dir
    model_path = args.object_model_path
    save_dir = args.save_dir
    time_window = args.time_window
    
    model_path = os.path.join(model_path, "saved_model")
    
    # Create folder for resample sensor data
    resample_save_folder_name = time_window + " Resampled"
    resample_save_folder_path = os.path.join(save_dir, resample_save_folder_name)
    os.makedirs(resample_save_folder_path, exist_ok=True)
    
    # Create folder for merged data with image folder for each object detection model used
    model_path_norm = os.path.normpath(model_path)
    object_model_name = model_path_norm.split(os.sep)[-2]
    merged_save_folder_path = os.path.join(save_dir, object_model_name)
    os.makedirs(merged_save_folder_path, exist_ok=True)
    
    # For this data we need to read sensor and emtion tag then merge them together

    sensor_path = os.path.join(data_path, "sensor")
    emotion_tag_path = os.path.join(data_path, "emotion_tags")
    
    # Read sensor data to df
    MNR_sensor_data = combine_sensor_data_to_df(sensor_path)
    # Read tag to df
    MNR_emotion_tags_data = combine_emotion_tags_to_df(emotion_tag_path)
    
    
    # Process the whole data
    # MNR_sensor_data_raw, MNR_sensor_data_processed, MNR_whole_data_labels = process_data(MNR_sensor_data)
    
    # Resample sensor data based on a time-window
    MNR_sensor_data_resampled = resample_data(MNR_sensor_data, time_window)
    
    # Process the resampled sensor data
    print("[*] Processing resampled sensor data  ....")
    MNR_sensor_data_resampled_raw, MNR_sensor_data_resampled_processed, MNR_sensor_data_resampled_labels = process_sensor_data(MNR_sensor_data_resampled)
    
    print(f"Resampled data size {MNR_sensor_data_resampled_raw.shape}")
    print(f"Resampled data processed size {MNR_sensor_data_resampled_processed.shape}")
    print(f"Resampled data label size {MNR_sensor_data_resampled_labels.shape}")
    
    # Save the processed data to csv file
    MNR_sensor_data_raw_file_name = os.path.join(resample_save_folder_path, "MNR_sensor_data_resampled_raw.csv")
    MNR_sensor_data_processed_file_name = os.path.join(resample_save_folder_path, "MNR_sensor_data_resampled_processed.csv")
    MNR_sensor_data_labels_file_name = os.path.join(resample_save_folder_path, "MNR_sensor_data_resampled_labels.csv")
    
    print("[*] Saving resampled sensor data...")
    MNR_sensor_data_resampled_raw.to_csv(MNR_sensor_data_raw_file_name, index=False, sep=',', header=True)
    MNR_sensor_data_resampled_processed.to_csv(MNR_sensor_data_processed_file_name, index=False, sep=',', header=True)
    MNR_sensor_data_resampled_labels.to_csv(MNR_sensor_data_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
    
    # Merge sensor to tags based on time column
    print("[*] Merging sensor data to emotion tags...")
    MNR_merged_data = merge_sensor_to_tag(MNR_sensor_data, MNR_emotion_tags_data, on="time")
    
    # Process the merged data
    print("[*] Processing merged data without image features...")
    MNR_merged_data_no_image_raw, MNR_merged_data_no_image_processed, MNR_merged_data_no_image_labels = process_merged_data_no_image(MNR_merged_data)
    print("Done")
    
    print(f"Merged data without image features size {MNR_merged_data_no_image_raw.shape}")
    print(f"Merged data without image features processed size {MNR_merged_data_no_image_processed.shape}")
    print(f"Merged labels without image size {MNR_merged_data_no_image_labels.shape}")
    
    # Save the processed data to csv file
    MNR_merged_data_no_image_raw_file_name = os.path.join(save_dir, "MNR_merged_data_no_image_raw.csv")
    MNR_merged_data_no_image_processed_file_name = os.path.join(save_dir, "MNR_merged_data_no_image_processed.csv")
    MNR_merged_data_no_image_labels_file_name = os.path.join(save_dir, "MNR_merged_data_no_image_labels.csv")
    print("[*] Saving merged data without image features....")
    MNR_merged_data_no_image_raw.to_csv(MNR_merged_data_no_image_raw_file_name, index=False, sep=',', header=True)
    MNR_merged_data_no_image_processed.to_csv(MNR_merged_data_no_image_processed_file_name, index=False, sep=',', header=True)
    MNR_merged_data_no_image_labels.to_csv(MNR_merged_data_no_image_labels_file_name, index=False, sep=',', header=True)
    print("Done")
    
    
    # Process the merged data with image features
    print("[*] Processing merged data without image features...")
    MNR_merged_data_raw, MNR_merged_data_processed, MNR_merged_data_labels = process_merged_data(MNR_merged_data, model_path)
    print("Done")
    
    print(f"Merged data with image features size {MNR_merged_data_raw.shape}")
    print(f"Merged data  with image features processed size {MNR_merged_data_processed.shape}")
    print(f"Merged data with image features labels size {MNR_merged_data_labels.shape}")
    
    
    MNR_merged_data_raw_file_name = os.path.join(merged_save_folder_path, "MNR_merged_data_raw.csv")
    MNR_merged_data_processed_file_name = os.path.join(merged_save_folder_path, "MNR_merged_data_processed.csv")
    MNR_merged_data_labels_file_name = os.path.join(merged_save_folder_path, "MNR_merged_data_labels.csv")
    
    # Save to csv files
    print("[*] Saving merged data with image features...")
    MNR_merged_data_raw.to_csv(MNR_merged_data_raw_file_name, index=False, sep=',', header=True)
    MNR_merged_data_processed.to_csv(MNR_merged_data_processed_file_name, index=False, sep=',', header=True)
    MNR_merged_data_labels.to_csv(MNR_merged_data_labels_file_name, index=False, sep=',', header=True)
    print("Done")

if __name__ == "__main__":
    main()
    