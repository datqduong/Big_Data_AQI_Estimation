# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:57:43 2020

@author: dat18
"""
import pandas as pd
import os
from datetime import datetime
from dateutil import rrule

def fah_to_cels(temp_fah):
    return (temp_fah - 32) * 5 / 9

# def combineImageAndSensorFeatures(sensor_data_raw, sensor_data_processed, data_processed_dir):
#     image_features_with_labels = pd.read_csv(os.path.join(data_processed_dir, "image_features_with_labels_sorted.csv"), index_col = 0)
#     image_names = image_features_with_labels.image_path.str.split('/').str[-1]
#     image_features_with_labels.insert(1, 'image', image_names)
    
#     sensor_data_processed['image'] = sensor_data_raw.image
    
#     #%% Join sensor features with image features
#     image_features_with_labels_dropped = image_features_with_labels.drop(columns=['image_path', 'Timestamp', 'day', 'month', 'pm25', 'aqi', 'aqi_rank'])
#     combined_features = sensor_data_processed.join(image_features_with_labels_dropped.set_index('image'), on='image')
    
#     combine_features_dropped = combined_features.drop(columns=['image'])
#     # timestamp_col = combine_features_dropped.pop("Timestamp")
#     # combine_features_dropped.insert(0, timestamp_col.name, timestamp_col)
#     #%% Save as csv
#     print("Saving the combined (sensor data + image) features...")
#     combine_features_dropped.to_csv(os.path.join(data_processed_dir, "combined_features.csv"), index=True, sep=',', header=True)
#     print("Done!")
#     return combine_features_dropped

# =============================================================================
# def combineGlobalFeatures(sensor_data_raw, sensor_data_processed, data_processed_dir):
#     # sensor_data_raw['timestamp'] = pd.to_datetime(sensor_data_raw['timestamp'], dayfirst = True)
#     
#     # combined_features = pd.read_csv(os.path.join(data_processed_dir, "combined_features.csv"), index_col = 0)
#     combined_features = combineImageAndSensorFeatures(sensor_data_raw, sensor_data_processed)
#     
#     # combined_features.insert(0, "Time", sensor_data_raw['timestamp'])
#     # sensor_data_processed.insert(0, 'Timestamp', sensor_data_raw['timestamp'])
#     
#     #%% Combine with global weather data of February 28 and March 6 2020
#     global_data_dir = "../../Data/Data HCM GAQD/Weather Data Daily"
#     from_date = datetime(2020, 2, 28)
#     to_date = datetime(2020, 3, 6)
#     
#     global_weather_df = pd.DataFrame()
#     # Loop for every day from "from_date" to "to_date" read the weather data and add the date to the "Time" column since it only has time data
#     for dt in rrule.rrule(rrule.DAILY, dtstart = from_date, until = to_date):
#         day, month, year = dt.day, dt.month, dt.year
#         # weather_file = os.path.join(global_data_dir, "Weather Data WUnderground 2020-{}-{}.csv".format(month, day))
#         # weather_files.append(weather_file)
#         global_weather = pd.read_csv(os.path.join(global_data_dir, "Weather Data WUnderground 2020-{}-{}.csv".format(month, day)), index_col = 0)
#         # Add date to "Time" column of the dataframe
#         global_weather['Time'] = dt.date().strftime("%Y-%m-%d") + ' ' + global_weather['Time']
#         # Convert string to datetime type
#         global_weather['Time'] = pd.to_datetime(global_weather['Time'])
#         global_weather.rename(columns={"Time": "Timestamp"}, inplace=True)
#         
#         global_weather['Temperature'] = global_weather['Temperature'].apply(fah_to_cels)
#         global_weather['Dew Point'] = global_weather['Dew Point'].apply(fah_to_cels)
#         # Combine all data together
#         global_weather_df = global_weather_df.append(global_weather, ignore_index=True)
#     
#     # Merge the features of global_weather_df to the sensor+image data based on the closest time reference from sensor+image data to global weather backwards
#     # E.g: If sensor time is: 7:00 AM, find the closest time before 7:00 AM (which can be 6:50 AM, or less if any closest is available) of weather data and merge the features to it.
#     combined_features_with_weather_data = pd.merge_asof(combined_features, global_weather_df, on='Timestamp')
#     
#     
#     #%% Save as csv
#     print("Saving the combined feature with global weather calculated features...")
#     combined_features_with_weather_data.to_csv(os.path.join(data_processed_dir, "combined_features_plus_global_weather.csv"), index=True, sep=',', header=True)
#     print("Done!")
#     
#     return combined_features_with_weather_data
# =============================================================================

def combinePublicWeather(data_to_combine, data_save_path):
    # Combine global data to the dataframe
    
    print("[*] Combining weather data...")
    #%% Combine sensor with global weather data
    global_data_dir = "../../Data/Data HCM GAQD/Weather Data Daily"
    from_date = data_to_combine.timestamp.iloc[0].date()
    to_date = data_to_combine.timestamp.iloc[-1].date()
    
    global_weather_df = pd.DataFrame()
    # Loop for every day from "from_date" to "to_date" read the weather data and add the date to the "Time" column since it only has time data
    for dt in rrule.rrule(rrule.DAILY, dtstart = from_date, until = to_date):
        day, month = dt.day, dt.month
        # weather_file = os.path.join(global_data_dir, "Weather Data WUnderground 2020-{}-{}.csv".format(month, day))
        # weather_files.append(weather_file)
        global_weather = pd.read_csv(os.path.join(global_data_dir, "Weather Data WUnderground 2020-{}-{}.csv".format(month, day)), index_col = 0)
        
        # Add date to "Time" column of the dataframe
        global_weather['Time'] = dt.date().strftime("%Y-%m-%d") + ' ' + global_weather['Time']
        # Convert string to datetime type
        global_weather['Time'] = pd.to_datetime(global_weather['Time'])
        global_weather.rename(columns={"Time": "timestamp"}, inplace=True)
        
        global_weather['Temperature'] = global_weather['Temperature'].apply(fah_to_cels)
        global_weather['Dew Point'] = global_weather['Dew Point'].apply(fah_to_cels)
        # Combine all data together
        global_weather_df = global_weather_df.append(global_weather, ignore_index=True)
        
    # Merge the features of global_weather_df to the data based on the closest time reference from sensor data to global weather backwards
    # E.g: If sensor time is: 7:00 AM, find the closest time before 7:00 AM (which can be 6:50 AM, or less if any closest is available) of weather data and merge the features to it.
    
    data_combined = pd.merge_asof(data_to_combine, global_weather_df, on='timestamp')
    
    print("[*] Saving the data combined with weather features...")
    data_combined.to_csv(data_save_path, index=False, sep=',', header=True)
    print("Done")
    return data_combined

def combinePublicWeather_(data_to_combine, data_save_path):
    # Combine global data to the dataframe
    
    print("[*] Combining weather data...")
    #%% Combine sensor with global weather data
    global_data_dir = "../../Data/Data MediaEval 2019 GAQD/Weather Data Daily"
    from_date = data_to_combine.timestamp.iloc[0].date()
    to_date = data_to_combine.timestamp.iloc[-1].date()
    
    global_weather_df = pd.DataFrame()
    # Loop for every day from "from_date" to "to_date" read the weather data and add the date to the "Time" column since it only has time data
    for dt in rrule.rrule(rrule.DAILY, dtstart = from_date, until = to_date):
        day, month, year = dt.day, dt.month, dt.year        
        # weather_file = os.path.join(global_data_dir, "Weather Data WUnderground 2020-{}-{}.csv".format(month, day))
        # weather_files.append(weather_file)
        global_weather = pd.read_csv(os.path.join(global_data_dir, "Weather Data WUnderground {}-{}-{}.csv".format(year, month, day)), index_col = 0)
        
        # Add date to "Time" column of the dataframe
        global_weather['Time'] = dt.date().strftime("%Y-%m-%d") + ' ' + global_weather['Time']
        # Convert string to datetime type
        global_weather['Time'] = pd.to_datetime(global_weather['Time'])
        global_weather.rename(columns={"Time": "timestamp"}, inplace=True)
        
        global_weather['Temperature'] = global_weather['Temperature'].apply(fah_to_cels)
        global_weather['Dew Point'] = global_weather['Dew Point'].apply(fah_to_cels)
        # Combine all data together
        global_weather_df = global_weather_df.append(global_weather, ignore_index=True)
        
    # Merge the features of global_weather_df to the data based on the closest time reference from sensor data to global weather backwards
    # E.g: If sensor time is: 7:00 AM, find the closest time before 7:00 AM (which can be 6:50 AM, or less if any closest is available) of weather data and merge the features to it.
    
    data_combined = pd.merge_asof(data_to_combine, global_weather_df, on='timestamp')
    
    print("[*] Saving the data combined with weather features...")
    data_combined.to_csv(data_save_path, index=False, sep=',', header=True)
    print("Done")
    return data_combined
    
# sensor_data = combine_data(data_path)
# sensor_data_raw, sensor_data_processed, labels_data = process_data(sensor_data)

# #%%
# image_features_with_labels = pd.read_csv(os.path.join(data_processed_dir, "image_features_with_labels.csv"), index_col = 0)
# image_names = image_features_with_labels.image_path.str.split('/').str[-1]
# image_features_with_labels.insert(1, 'image', image_names)

# sensor_data_processed['image'] = sensor_data_raw.image

# #%% Join sensor features with image features
# image_features_with_labels_dropped = image_features_with_labels.drop(columns=['image_path', 'pm25', 'aqi', 'aqi_rank'])
# combined_features = sensor_data_processed.join(image_features_with_labels_dropped.set_index('image'), on='image')

# combine_features_dropped = combined_features.drop(columns=['image'])
# #%% Save as csv
# combine_features_dropped.to_csv(os.path.join(data_processed_dir, "combined_features.csv"), index=True, sep=',', header=True)