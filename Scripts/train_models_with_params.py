# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:36:55 2020

@author: dat18
"""
# Import utilities
import os
import argparse
from utils import display_Results_One_Pol
from models.read_processed_data_csv import get_train_test_data

# Import regression models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def parse_arguments():
    MODELS_CHOICES = ["SVM", "Random Forest", "Catboost", "XGBoost", "LightGBM", "All"]
    FEATURE_TYPES = ["Sensor", "Sensor+PW", "Tag+Sensor", "Tag+Sensor+PW", "Tag+Sensor+Image", "Tag+Sensor+Image+PW"]
    PREDS_CHOICES = ["pm25", "pm10", "o3", "co", "so2", "no2", "aqi"]
    
    ap = argparse.ArgumentParser()
    arg = ap.add_argument
    arg("-d", "--data_processed_dir", required=True, type=str, help="Directory of the train and test csv data (including random split folders)")
    
    arg("-f", "--feature_type", required=True, type=str, help="Type of data split", choices= FEATURE_TYPES)
    arg("-mc", "--model_choice", required=True, type=str, help="Model name to use, use 'All' to use all available models", choices = MODELS_CHOICES)
    
    arg("-p", "--pollutant_to_predict", required=True, type=str, help="Specify name of pollutant to predict", choices = PREDS_CHOICES)
    
    arg("-om", '--object_model_name', type=str, help="Name of the object model used to extact Image features", choices=["SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)", "EfficientDet D7 1536x1536"])
    
    arg("-ms", "--model_save_path", required=True, type=str, help="Path to save the model")
    arg("-rs", "--results_save_path", required=True, type=str, help="Path to save output result")
    args = ap.parse_args()

    if "Image" in args.feature_type and args.object_model_name is None:
        ap.error("--object_model_name argument is required when using this feature type")
        
    assert os.path.exists(os.path.realpath(os.path.expanduser(args.data_processed_dir))),\
            f"The directory '{args.data_processed_dir}' doesn't exist!"
    return args

def fit_and_evaluate(model, inputs, model_name):
    X_train_random_split = inputs.X_train_random_split
    y_train_random_split = inputs.y_train_random_split
    X_test_random_split = inputs.X_test_random_split
    y_test_random_split = inputs.y_test_random_split
    
    # model_save_path = inputs.model_save_path
    results_save_path = inputs.results_save_path
    pol_pred_name = inputs.pollutant_to_predict
    
    print("[*] Fitting model {}...".format(model_name))
    model.fit(X_train_random_split, y_train_random_split[pol_pred_name])
    print("Done")
    
    print("[*] Evaluating with Test set...")
    preds = model.predict(X_test_random_split)
    print("[*] Done")
    
    save_path = os.path.join(results_save_path, "Random Split")
    os.makedirs(save_path, exist_ok=True)
    filePath = os.path.join(save_path, model_name + " Results.txt")

    print("[*] Showing evaluation metrics results")
    # display_Results(y_test_random_split, preds, writeFile = True, fPath = filePath, modelName = model_name)
    y_true_pol = y_test_random_split[pol_pred_name]
    display_Results_One_Pol(y_true_pol, preds, writeFile = True, fPath = filePath, modelName= model_name)
    
def main(args):
    print("Args input entered: {}".format(args))
    
    feature_type = args.feature_type
    data_processed_dir = args.data_processed_dir
    object_model_name = args.object_model_name
    
    models_map = {
         "SVM": SVR,
         "Random Forest": RandomForestRegressor,         
         "Catboost": CatBoostRegressor,
         "XGBoost": XGBRegressor, 
         "LightGBM": LGBMRegressor
        }
    
    SVR_params = {
                    'kernel': 'rbf',
                    'degree': 3,
                    'gamma': 'scale',
                    'coef0': 0.0,
                    'tol': 1e-3,
                    'C': 100,
                    'epsilon': 0.5,
                    'shrinking': True,
                    'cache_size': 200,
                    'verbose': False,
                    'max_iter': -1
                 }
    
    RF_params = { 
                    'n_estimators': 1000,
                    'criterion': 'mse',
                    'max_depth': None,
                    'min_samples_split': 4,
                    'min_samples_leaf': 1,
                    'min_weight_fraction_leaf':0.0,
                    'max_features': 'sqrt',
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0.0,
                    'bootstrap': True,
                    'oob_score': True,
                    'n_jobs': None,
                    'random_state': 24,
                    'verbose': 0,
                    'warm_start': False,
                    'ccp_alpha': 0.0
                }
    
    CB_params = {
                    'loss_function': 'RMSE',
                    'n_estimators': 1000,
                    'learning_rate': 0.03,
                    'depth': 10,
                    'l2_leaf_reg': 3.0,
                    'eval_metric': 'RMSE',
                    'random_seed': 24,
                    'verbose': 0,
                }
    
    XGB_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 1000,
                    "learning_rate": 0.05,
                    'max_depth': 10,
                    'min_child_weight': 1,
                    'verbosity': 0,
                    'random_state': 24   
                 }
    
    LGBM_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': 24
                  }
    
    
    params_map = {                    
                     "SVM": SVR_params,
                     "Random Forest": RF_params,
                     "Catboost": CB_params,
                     "XGBoost": XGB_params,
                     "LightGBM": LGBM_params
                 }       
    
    
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


    args.model_save_path = os.path.join(args.model_save_path, data_folder_name)
    args.results_save_path = os.path.join(args.results_save_path, data_folder_name)
    
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.results_save_path, exist_ok=True)
        
    random_split_dir = os.path.join(data_processed_dir, data_folder_name, 'Random split')
    
    X_train_random_split, y_train_random_split, X_test_random_split, y_test_random_split = get_train_test_data(random_split_dir, train_data_standardized_name, train_label_name, test_data_standardized_name, test_label_name)
    
    print(f"X_train size {X_train_random_split.shape}")
    print(f"y_train size {y_train_random_split.shape}")
    print(f"X_test size {X_test_random_split.shape}")
    print(f"y_test size {y_test_random_split.shape}")
    
    print(X_train_random_split.isnull().sum())
    print(y_train_random_split.isnull().sum())
    print(X_test_random_split.isnull().sum())
    print(y_test_random_split.isnull().sum())
    
    
    model_choice = args.model_choice
    
    inputs = args
    inputs.X_train_random_split = X_train_random_split
    inputs.y_train_random_split = y_train_random_split
    inputs.X_test_random_split = X_test_random_split
    inputs.y_test_random_split = y_test_random_split
   
    
    if model_choice != "All":       
        print(f"[*] Running model {model_choice} using pre-defined params")
        params = params_map[model_choice]
        model = models_map[model_choice](**params)        
        fit_and_evaluate(model, inputs, model_choice)      
    else:
        print("[*] Running all models...")
        for model_name in models_map:
            print(f"[*] Running model {model_name} using pre-defined params")
            params = params_map[model_name]
            model = models_map[model_name](**params)
            fit_and_evaluate(model, inputs, model_name)
        print("[*] Finished")
    
if __name__ == "__main__":
    args = parse_arguments()
            
    # log_dir = "output_logs/"
    # os.makedirs(log_dir, exist_ok=True)
    # now = datetime.now()
    # log_file_name = "output_" + now.strftime("%d-%m-%Y_%H-%M-%S") + ".txt"
    # sys.stdout = open(os.path.join(log_dir, log_file_name), "w")
    main(args)
    # sys.stdout.close()