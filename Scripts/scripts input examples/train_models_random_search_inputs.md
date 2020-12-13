## Train models using Randomized Search CV usage examples

Use `python train_models_random_search.py -h` to look at all arguments. Remember to set *--data_processed_dir, --model_save_path, and --results_save_path* to different folders for each Data set
Set --model_choice to "All" to fit all models using Randomized Search.

## Example usage on MNR data set

### Inputs to train SVM model on MNR dataset using Sensor features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```
### Inputs to train SVM model on MNR dataset using "Sensor+PW" features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor+PW" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```
### Inputs to train SVM model on MNR dataset using "Tag+Sensor" features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```
### Inputs to train SVM model on MNR dataset using "Tag+Sensor+PW" features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+PW" --model_choice "SVM" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```

**Note**

Scenarios with Image feature need to have one more argument *--object_model_name*

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image" features using object model SSD
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --model_choice "SVM" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image" features using object model EfficientDetD7
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --model_choice "SVM" --object_model_name "EfficientDet D7 1536x1536" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image+PW" features using object model SSD
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image+PW" --model_choice "SVM" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```

### Inputs to train SVM model on MNR dataset using "Tag+Sensor+Image+PW" features using object model EfficientDetD7
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image+PW" --model_choice "SVM" --object_model_name "EfficientDet D7 1536x1536" --model_save_path "../Saved Models/MNR 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR 30S Dataset/Randomized Search/"
```

## Example usage on MNR Air data set

### Inputs to train **All** model on MNR Air dataset using Sensor features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MNR Air (NoUV) Processed/" --feature_type "Sensor" --model_choice "All" --model_save_path "../Saved Models/MNR Air 30S Dataset/Randomized Search/" --results_save_path "../Results/MNR Air 30S Dataset/Randomized Search/"
```

## Example usage on MediaEval 2019 data set
### Inputs to train **All** model on MNR dataset using Sensor features
```bash
python train_models_random_search.py --data_processed_dir "../Data Processed/MediaEval2019 Processed/" --feature_type "Sensor" --model_choice "All" --model_save_path "../Saved Models/MediaEval2019 Dataset/Randomized Search/" --results_save_path "../Results/MediaEval2019 Dataset/Randomized Search/"
```
