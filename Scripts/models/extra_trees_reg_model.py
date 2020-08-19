# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:39:41 2020

@author: dat18
"""

#%%
import os
import numpy as np
#%%
from sklearn.ensemble import ExtraTreesRegressor
from utils import display_Results, save_Model, grid_search, save_best_params

def fit_model(inputs):
    
    X_train_random_split = inputs.X_train_random_split
    y_train_random_split = inputs.y_train_random_split
    X_test_random_split = inputs.X_test_random_split
    y_test_random_split = inputs.y_test_random_split
    X_train_hist_split = inputs.X_train_hist_split
    y_train_hist_split = inputs.y_train_hist_split
    X_test_hist_split = inputs.X_test_hist_split
    y_test_hist_split = inputs.y_test_hist_split
    
    params = {
        'n_estimators': 1000,
        'criterion': 'mse',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': None,
        'random_state': 24,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0.0,
        'max_samples': None,
        }
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 8, 16]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8, 16]
    
    # Define param space to search
    param_space = {                    
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,                
                   }
    
    model_name = "Extra Trees Regression"
    
    model_save_path = inputs.model_save_path
    results_save_path = inputs.results_save_path
    
    print(model_name)
    ###################### Random Split Training ###########################
    random_split_model = ExtraTreesRegressor(**params)
    random_split_model_best = grid_search(random_split_model, param_space)
    
    print("Fitting model using Random Split Data...")
    # Single variable
    random_split_model_best.fit(X_train_random_split, y_train_random_split.pm25)
    # Multi-variable
    # model.fit(X_train, y_train.drop(columns=['aqi', 'aqi_rank']))
    print("Done!")
    print("Best params: {}".format(random_split_model_best.best_params_))
    
    
    #%% Evaluate model
    print("Evaluating with Test set")
    preds_random_split = random_split_model_best.predict(X_test_random_split)
    print("Done!")
    
    #%% Random Split Results
    save_path = os.path.join(results_save_path, "Random Split")
    os.makedirs(save_path, exist_ok=True)
    filePath = os.path.join(save_path, model_name + " Results.txt")
    bestparamPath = os.path.join(save_path, model_name + " Best params.txt")
    print("Saving best params...")
    save_best_params(random_split_model_best, bestparamPath)
    print("Done!")
    print("Showing result for Random Split")
    display_Results(y_test_random_split, preds_random_split, writeFile = True, fPath = filePath, modelName = model_name)
    
    #%% Save model
    model_save_name = model_name + " Random Split Model.pickle"
    save_Model(random_split_model_best, model_save_name, model_save_path)
    
    #%%#######################################################################
    ###################### Historical Split Training #########################
    hist_split_model = ExtraTreesRegressor(**params)
    hist_split_model_best = grid_search(hist_split_model, param_space)
    # Single variable
    print("Fitting model using Historical Split Data...")
    hist_split_model_best.fit(X_train_hist_split, y_train_hist_split.pm25)
    # Multi-variable
    # model.fit(X_train, y_train.drop(columns=['aqi', 'aqi_rank']))
    print("Done!")
    print("Best params: {}".format(hist_split_model_best.best_params_))
    
    #%% Evaluate model
    print("Evaluating with Test set")
    preds_hist_split = hist_split_model_best.predict(X_test_hist_split)
    print("Done!")
    
    #%% Historical Split Results
    save_path = os.path.join(results_save_path, "Historical Split")
    os.makedirs(save_path, exist_ok=True)
    filePath = os.path.join(save_path, model_name + " Results.txt")
    bestparamPath = os.path.join(save_path, model_name + " Best params.txt")    
    print("Saving best params...")
    save_best_params(hist_split_model_best, bestparamPath)
    print("Done!")
    print("Showing result for Historical Split")
    display_Results(y_test_hist_split, preds_hist_split, writeFile = True, fPath = filePath, modelName = model_name)
    
    #%% Save model
    model_save_name = model_name + " Historical Split Model.pickle"
    save_Model(hist_split_model_best, model_save_name, model_save_path)
