## Split data usage examples

### Example commands for MNR Data split
```python split_data.py [args]```. Specify  one of the following options for [args]:
```bash
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor"
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Sensor+PW"
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor"
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+PW"

--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)"
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image" --object_model_name "EfficientDet D7 1536x1536"

--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image+PW" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)"
--dataset_type "MNR" --data_processed_dir "../Data Processed/MNR Processed/" --feature_type "Tag+Sensor+Image+PW" --object_model_name "EfficientDet D7 1536x1536"
```
### Example commands for MNR Air Data split
```python split_data.py [args]```. Specify one of the following options for [args]:
```bash
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Sensor"
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Sensor+PW"
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor"
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor+PW"

--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor+Image" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)"
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor+Image" --object_model_name "EfficientDet D7 1536x1536"

--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor+Image+PW" --object_model_name "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)"
--dataset_type "MNR_AIR" --data_processed_dir "../Data Processed/MNR Air Processed/" --feature_type "Tag+Sensor+Image+PW" --object_model_name "EfficientDet D7 1536x1536"
```