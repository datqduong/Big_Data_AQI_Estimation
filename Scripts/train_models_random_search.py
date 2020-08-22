# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:32:47 2020

@author: dat18
"""
import os
import argparse
from models.read_processed_data_csv import get_train_test_data
from scipy.stats import loguniform, uniform, randint

# Import regression models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from utils import display_Results, save_best_model, random_search, save_best_params

def parse_arguments():

    MODELS_CHOICES = ["SVM", "Random Forest", "Catboost", "XGBoost", "LightGBM", "All"]
    FEATURE_TYPES = ["Sensor", "Sensor+PW", "Tag+Sensor", "Tag+Sensor+PW", "Tag+Sensor+Image", "Tag+Sensor+Image+PW"]
    
    ap = argparse.ArgumentParser()
    arg = ap.add_argument
    arg("--data_processed_dir", required=True, type=str, help="Directory of the train and test csv data (including random split folders)")
    
    arg("--feature_type", required=True, type=str, help="Type of data split", choices= FEATURE_TYPES)
    arg("--model_choice", required=True, type=str, help="Model name to use, use 'All' to use all available models", choices = MODELS_CHOICES)
    arg('--object_model_name', type=str, help="Name of the object model used to extact Image features", choices=["SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)", "EfficientDet D7 1536x1536"])
    
    arg("--model_save_path", required=True, type=str, help="Path to save the model")
    arg("--results_save_path", required=True, type=str, help="Path to save output result")
    args = ap.parse_args()

    if "Image" in args.feature_type and args.object_model_name is None:
        ap.error("--object_model_name argument is required when using this feature type")
        
    assert os.path.exists(os.path.realpath(os.path.expanduser(args.data_processed_dir))),\
            f"The directory '{args.data_processed_dir}' doesn't exist!"
    return args

def search_and_evaluate(model, inputs, model_name):
    
    X_train_random_split = inputs.X_train_random_split
    y_train_random_split = inputs.y_train_random_split
    X_test_random_split = inputs.X_test_random_split
    y_test_random_split = inputs.y_test_random_split
    
    model_save_path = inputs.model_save_path
    results_save_path = inputs.results_save_path
    
    param_space = inputs.params_map[model_name]
    search = random_search(model, param_space)
    print("[*] Performing randomized search...")
    search.fit(X_train_random_split, y_train_random_split.aqi)
    
    print("Done!")
    
    # Save best params
    print("Best params: {}".format(search.best_params_))
    save_path = os.path.join(results_save_path, "Random Split")
    os.makedirs(save_path, exist_ok=True)
    bestparamPath = os.path.join(save_path, model_name + " Best params.txt")
    print("[*] Saving best params...")
    save_best_params(search, bestparamPath)
    print("Done!")
    
    #%% Evaluate model
    print("[*] Evaluating on hold out test set...")
    preds_random_split = search.predict(X_test_random_split)
    print("Done!")
    
    #%% Display hold out test set results
    filePath = os.path.join(save_path, model_name + " Results.txt")
    print("[*] Showing result for Random Split")
    display_Results(y_test_random_split, preds_random_split, writeFile = True, fPath = filePath, modelName = model_name)
    
    #%% Save best model
    model_save_name = model_name + " Random Split Best Model.pkl"
    save_best_model(search, model_save_name, model_save_path)
    
    
def main():
    args = parse_arguments()
    print("Args input entered: {}".format(args))
    
    feature_type = args.feature_type
    data_processed_dir = args.data_processed_dir
    object_model_name = args.object_model_name
    
    if feature_type == "Sensor":
        data_folder_name = "Sensor Features"
        
        train_data_standardized_name = "train_data_sensor_standardized.csv"
        train_label_name  = "train_label_sensor.csv"
        test_data_standardized_name = "test_data_sensor_standardized.csv"
        test_label_name = "test_label_sensor.csv"
        
    elif feature_type == "Sensor+PW":
        data_folder_name = "Sensor+PW Features"
        
        train_data_standardized_name = "train_data_sensor_pw_standardized.csv"
        train_label_name = "train_label_sensor_pw.csv"
        test_data_standardized_name = "test_data_sensor_pw_standardized.csv"
        test_label_name = "test_label_sensor_pw.csv"
    
    elif feature_type == "Tag+Sensor":
        data_folder_name = "Tag+Sensor"
        
        train_data_standardized_name = "train_data_tag_sensor_standardized.csv"
        train_label_name = "train_label_tag_sensor.csv"
        test_data_standardized_name = "test_data_tag_sensor_standardized.csv"
        test_label_name = "test_label_tag_sensor.csv"
    
    elif feature_type == "Tag+Sensor+PW":
        data_folder_name = "Tag+Sensor+PW"

        train_data_standardized_name = "train_data_tag_sensor_pw_standardized.csv"
        train_label_name = "train_label_tag_sensor_pw.csv"
        test_data_standardized_name = "test_data_tag_sensor_pw_standardized.csv"
        test_label_name = "test_label_tag_sensor_pw.csv"
        
    elif feature_type == "Tag+Sensor+Image":
        data_folder_name = "Tag+Sensor+Image" + "_" + object_model_name

        train_data_standardized_name = "train_data_tag_sensor_image_standardized.csv"
        train_label_name = "train_label_tag_sensor_image.csv"
        test_data_standardized_name = "test_data_tag_sensor_image_standardized.csv"
        test_label_name = "test_label_tag_sensor_image.csv"

        
    elif feature_type == "Tag+Sensor+Image+PW":
        data_folder_name = "Tag+Sensor+Image+PW" + "_" + object_model_name
        
        train_data_standardized_name = "train_data_tag_sensor_image_pw_standardized.csv"
        train_label_name = "train_label_tag_sensor_image_pw.csv"
        test_data_standardized_name = "test_data_tag_sensor_image_pw_standardized.csv"
        test_label_name = "test_label_tag_sensor_image_pw.csv"


    models_map = {
                     "SVM": SVR(),
                     "Random Forest": RandomForestRegressor(random_state=24),
                     "Catboost": CatBoostRegressor(objective="RMSE", random_state=24),
                     "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=24, verbosity=0),
                     "LightGBM": LGBMRegressor(objective='regression', random_state=24, silent=True)
                 }
    
    SVR_param_space = {
                        'kernel': ['rbf'],
                        'C': loguniform(1e-3, 1e3),
                        'degree': randint(1, 4),
                        'gamma': ['scale', 'auto'],
                        'epsilon': loguniform(1e-1, 5e-1)
                       }
    
    RF_param_space = {
                        'n_estimators': randint(1e2, 1e3),    
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': randint(5, 20),
                        'min_samples_split': randint(2, 10),
                        'min_samples_leaf': randint(1, 10)                    
                     }
    
    XGB_param_space = {                         
                         'n_estimators': randint(1e2, 1e3),                          
                         'gamma': loguniform(1e-3, 1),                         
                         'max_depth': randint(5, 20),
                         'min_child_weight': randint(1, 10),
                         'reg_lambda ': loguniform(1e-3, 1),
                         'reg_alpha ': loguniform(1e-3, 1),
                         'learning_rate': loguniform(1e-4, 1e-1)                        
                      }
    
    LGBM_param_space = { 
                         'boosting_type': ['gbdt'],                        
                         'n_estimators': randint(1e2, 1e3),                        
                         'max_depth': randint(5, 30),
                         'learning_rate': loguniform(1e-4, 1e-1),
                         'num_leaves': randint(2, 256),
                         'min_child_weight': uniform(1e-4, 1e-2),
                         'min_child_samples': randint(10, 30),
                         'reg_lambda ': loguniform(1e-3, 1),
                         'reg_alpha ': loguniform(1e-3, 1)                          
                        }
    
    CB_param_space = {                                                
                        'n_estimators': randint(1e2, 1e3),                        
                        'learning_rate': loguniform(1e-4, 1e-1),
                        'depth': randint(4, 12),
                        'l2_leaf_reg': randint(1, 10),
                        'rsm': uniform(1e-3, 9e-1),
                        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],                        
                     }
    
    params_map = {
                     "SVM": SVR_param_space,
                     "Random Forest": RF_param_space,
                     "Catboost": CB_param_space,
                     "XGBoost": XGB_param_space,
                     "LightGBM": LGBM_param_space
                 }
            
    args.model_save_path = os.path.join(args.model_save_path, data_folder_name)
    args.results_save_path = os.path.join(args.results_save_path, data_folder_name)
    
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.results_save_path, exist_ok=True)
    
    random_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Random split')
    
    X_train_random_split, y_train_random_split, X_test_random_split, y_test_random_split = get_train_test_data(random_split_dir, train_data_standardized_name, train_label_name, test_data_standardized_name, test_label_name)
    
    model_choice = args.model_choice
    
    inputs = args
    inputs.X_train_random_split = X_train_random_split
    inputs.y_train_random_split = y_train_random_split
    inputs.X_test_random_split = X_test_random_split
    inputs.y_test_random_split = y_test_random_split
    inputs.params_map = params_map
    
    if model_choice != "All":
        print(f"[*] Searching and evaluating model {model_choice}")
        model = models_map[model_choice]
        search_and_evaluate(model, inputs, model_choice)
        print("Done")
    elif model_choice == "All":
        print("[*] Running all models...")
        for model_name in models_map:
            print(f"[*] Searching and evaluating model {model_name}")
            model = models_map[model_name]
            search_and_evaluate(model, inputs, model_name)
        print("Done!")
    
if __name__ == "__main__":
    
    main()


