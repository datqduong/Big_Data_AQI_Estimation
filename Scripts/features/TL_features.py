# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:51:41 2020

@author: dat18
"""
import numpy as np
import pandas as pd

def get_part_of_day(time):
    if (time >= 5) and (time < 7):
        return "Early Morning"
    elif (time >= 7) and (time < 12):
        return "Morning"
    elif (time >= 12) and (time < 16):
        return "Afternoon"
    elif (time >= 16) and (time < 20):
        return "Evening"
    else:
        return "Late Night"
        
def is_RushHour(time):
    return "Rush hour" if (time >= 7 and time <= 9) or (time >= 16 and time <= 19) else "Not rush hour"

def haversine_distance(lat1, lon1, lat2, lon2):
    # Compute haversine distance between user location with "destination" (airport or city center)
    
    # Haversine formula
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2.0) * np.sin(dlat / 2.0) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) * np.sin(dlon / 2.0)
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
    
    return (c * r)

def distance_to_airport(location):
    # Tan Son Nhat Airport coordinates according to google maps
    TSN_airport_lat = 10.817996728
    TSN_airport_lon = 106.651164062
    
    lat_col = location['lat']
    lon_col = location['lon']
    
    distance = haversine_distance(lat_col, lon_col, TSN_airport_lat, TSN_airport_lon)
    return distance