# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:18:41 2020

@author: dat18
"""
#%%
import os

#%%
from sklearn.svm import SVR
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
        'kernel': 'rbf',
        'degree': 3,
        'gamma': 'scale',
        'coef0': 0.0,
        'tol': 1e-3,
        'C': 1.0,
        'epsilon': 0.1,
        'shrinking': True,
        'cache_size': 200,
        'verbose': False,
        'max_iter': -1
        }
    
    # Define param space to search
    param_space = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 1e-3, 1e-4],
                    'epsilon': [0.1, 0.2, 0.3, 0.5],
                    'kernel': ['rbf']
                   }
    print("Parameter space: {}".format(param_space))
    
    model_name = "Support Vector Regression"
    
    model_save_path = inputs.model_save_path
    results_save_path = inputs.results_save_path
    
    print(model_name)
    ###################### Random Split Training ###########################
    random_split_model = SVR(**params)
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
    hist_split_model = SVR(**params)
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