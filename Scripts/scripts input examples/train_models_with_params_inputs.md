## Train models with pre-defined params usage examples

Use `python train_models_with_params.py -h` to look at all arguments. Remember to set *--data_processed_dir, --model_save_path, and --results_save_path* to different folders for each Data set
Set --model_choice to "All" to fit all models on pre-defined params

## Example usage on MNR data set

### Inputs to train SVM model on MNR dataset using Sensor features
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```
### Inputs to train SVM model on MNR dataset using "Sensor+PW" features
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor+PW" --model_choice "SVM" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```
### Inputs to train SVM model on MNR dataset using "Tag+Sensor" features
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor" --model_choice "SVM" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```
### Inputs to train SVM model on MNR dataset using "Tag+Sensor+PW" features
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+PW" --model_choice "SVM" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

**Note**

Scenarios with Image feature need to have one more argument *--object_model_name*

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image" features using object model SSD
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --model_choice "SVM" --pollutant_to_predict "aqi" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image" features using object model EfficientDetD7
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --model_choice "SVM" --pollutant_to_predict "aqi" --object_model_name "EfficientDet D7 1536x1536" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image+PW" features
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image+PW" --model_choice "SVM" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MNR 30S Dataset/" --results_save_path "../Results/Test Params/MNR 30S Dataset"
```

## Example usage on MNR Air data set
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Air (NoUV) Processed/" --feature_type "Sensor" --model_choice "All" --pollutant_to_predict "aqi" --object_model_name "EfficientDet D7 1536x1536" --model_save_path "../Saved Models/Test Params/MNR Air 30S Dataset/" --results_save_path "../Results/Test Params/MNR Air 30S Dataset"
```

## Example usage on MediaEval 2019 data set
```bash
python train_models_with_params.py --data_processed_dir "../Data Processed/MNR Air (NoUV) Processed/" --feature_type "Sensor" --model_choice "All" --pollutant_to_predict "aqi" --model_save_path "../Saved Models/Test Params/MediaEval2019 Dataset/" --results_save_path "../Results/Test Params/MediaEval2019 Dataset"
```
