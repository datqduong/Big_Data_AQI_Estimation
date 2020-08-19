# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:19:37 2020

@author: dat18
"""
from Calculate_AQI import calculateAQI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os
import pickle
import json
from sklearn.model_selection import GridSearchCV
np.random.seed(24)

def resample_data(df, time_window):
    print(f"[*] Resampling data based on {time_window} time window")
    df_remove_dup = df.drop_duplicates('time', keep='first', inplace=False, ignore_index=True)
    resampled_data = pd.DataFrame()
    resampler = df_remove_dup.set_index('time').resample(time_window)
    resampled_groups = resampler.groups
    for key in resampled_groups:
        indices = resampler._get_index(key)
        # If there are values in the group bin
        if len(indices):
            # Pick a random value from the group
            indx_choice = [np.random.choice(indices)]
            # Get the picked value from df
            value_df = df_remove_dup.take(indx_choice)
            resampled_data = pd.concat([resampled_data, value_df])      
    print("Done")
    return resampled_data

def grid_search(estimator, param_space):
    grid_search_model = GridSearchCV(estimator = estimator, param_grid = param_space, cv=5, scoring = 'r2', verbose=0, n_jobs=-1)
    return grid_search_model
    
def save_best_params(estimator, save_path):
    with open(save_path, "w") as file:
        json.dump(estimator.best_params_, file, indent=4)
    
def display_Results(y_true, y_pred, plot=False, writeFile = False, **kwargs):
    # Single-variable score
    print("Regression Score for PM2.5 values")
    rmse = np.sqrt(mean_squared_error(y_true.pm25, y_pred))
    mae = mean_absolute_error(y_true.pm25, y_pred)
    r2 = r2_score(y_true.pm25, y_pred)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 score:", r2)
    
    # Multi-variable score
    # print("Regression Score for all pollutants:")
    # rmse = np.sqrt(mean_squared_error(y_true.drop(columns=['aqi', 'aqi_rank']), y_pred))
    # mae = mean_absolute_error(y_true.drop(columns=['aqi', 'aqi_rank']), y_pred)
    # r2 = r2_score(y_true.drop(columns=['aqi', 'aqi_rank']), y_pred)
    # print("RMSE:", rmse)
    # print("MAE:", mae)
    # print("R2 score:", r2)
    
    # Multi-variate predictions
    # preds_df = pd.DataFrame(data=y_pred, columns=y_true.columns[:-2])
    # Single-variate prediction
    y_pred_df = create_empty_pollutants_columns(y_pred, 'pm25')
    AQI_list_np, AQI_rank_np = calc_AQI_from_df(y_pred_df)
    y_pred_df['aqi'] = AQI_list_np
    y_pred_df['aqi_rank'] = AQI_rank_np
    
    print("Regression Score for AQI values")
    rmse_aqi = np.sqrt(mean_squared_error(y_true.aqi, y_pred_df.aqi))
    mae_aqi = mean_absolute_error(y_true.aqi, y_pred_df.aqi)
    r2_aqi = r2_score(y_true.aqi, y_pred_df.aqi)
    print("RMSE:", rmse_aqi)
    print("MAE:", mae_aqi)
    print("R2 score:", r2_aqi)
    
    print("Classification score for AQI rank")
    range_ = list(range(0, 6))
    accuracy = accuracy_score(y_true.aqi_rank, y_pred_df.aqi_rank) * 100
    f1 = f1_score(y_true.aqi_rank, y_pred_df.aqi_rank, labels=range_, average='weighted', zero_division=0) * 100
    print("Accuracy Score: {:.2f}%".format(accuracy))
    print("F1 score: {:.2f}%".format(f1))
    conf_matrix = confusion_matrix(y_true.aqi_rank, y_pred_df.aqi_rank, labels=range_)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    if plot:
        print("Plotting...")
        plt.figure()
        plot_confusion_matrix(conf_matrix, classes=range_)
        plt.show()
    
    if writeFile:
        model_name = kwargs['modelName']
        file_name = kwargs['fPath'].split('.txt')[0] + '.xlsx'
        print("Writing results to excel file...")
        reg_pm25_score_df = pd.DataFrame({"RMSE": rmse, "MAE": mae, "R2": r2}, index=pd.Series(model_name, name="Model name"))
        reg_aqi_score_df = pd.DataFrame({"RMSE": rmse_aqi, "MAE": mae_aqi, "R2": r2_aqi}, index=pd.Series(model_name, name="Model name"))
        aqi_rank_score_df = pd.DataFrame({"Accuarcy": accuracy, "F1": f1}, index=pd.Series(model_name, name="Model name"))
        
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        reg_pm25_score_df.to_excel(writer, sheet_name="PM2.5 prediction score", float_format = "%0.2f")
        reg_aqi_score_df.to_excel(writer, sheet_name="PM2.5 to AQI values score", float_format = "%0.2f")
        aqi_rank_score_df.to_excel(writer, sheet_name="PM2.5 to AQI rank score", float_format = "%0.2f")
        
        writer.save()
        
        print("Done!")
        print("Writing results to text file...")
        filePath = kwargs['fPath']
        save_folder = os.path.split(filePath)[0]
        os.makedirs(save_folder, exist_ok=True)
        with open(filePath, "w") as file:
            file.write(model_name + "\n")
            
            file.write("\tPM2.5 predictions score\n")
            file.write("\t* RMSE - {:.2f}\n".format(rmse))
            file.write("\t* MAE - {:.2f}\n".format(mae))
            file.write("\t* R2 - {:.2f}\n\n".format(r2))
            
            file.write("\tAQI values prediction score\n")
            file.write("\t* RMSE - {:.2f}\n".format(rmse_aqi))
            file.write("\t* MAE - {:.2f}\n".format(mae_aqi))
            file.write("\t* R2 - {:.2f}\n\n".format(r2_aqi))
            
            file.write("\tAQI rank prediction score\n")
            file.write("\t* Accuracy - {:.2f}\n".format(accuracy))
            file.write("\t* F1 score - {:.2f}\n\n".format(f1))
        print("Done!")

def save_Model(model, model_save_name, model_save_path):
    os.makedirs(model_save_path, exist_ok=True)
    print("Saving model...")
    filename = os.path.join(model_save_path, model_save_name)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("Done!")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
  
def calc_AQI_from_df(input_df):
    # Calculate AQI and AQI rank for input dataframe
    AQI_list = []
    AQI_rank = []
    pollutants = ['0x_8', '0x', 'PM10', 'PM25', 'CO', 'SO2', 'SO2_24', 'NO2']
    
    for row in input_df.to_numpy():
        myAQI = { key:value for key, value in zip(pollutants, row)}
        AQI_val = calculateAQI(myAQI).getAQI()
        AQI_rank_val = calculateAQI(myAQI).getAQIRank()
        AQI_list.append(AQI_val)
        AQI_rank.append(AQI_rank_val)
    
    AQI_list_np = np.array(AQI_list)
    AQI_rank_np = np.array(AQI_rank)
    
    # Return lists of AQIs and AQI_ranks
    return AQI_list_np, AQI_rank_np

def create_empty_pollutants_columns(preds, pollutants):
    preds_df = pd.DataFrame(data=preds, columns=[pollutants])
    # Add empty values for other pollutants
    empty_pollutant = ['o3_8', 'o3', 'pm10', 'co', 'so2', 'so2_24', 'no2']
    for col in empty_pollutant:
        preds_df[col] = 0
    preds_df= preds_df.reindex(columns=['o3_8', 'o3', 'pm10', 'pm25', 'co', 'so2', 'so2_24', 'no2'])
    return preds_df