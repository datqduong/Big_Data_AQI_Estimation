# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:07:52 2020

@author: dat18

"""
from tqdm import tqdm
import time
from PIL import Image
import numpy as np
import pandas as pd
import os

import tensorflow as tf
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
      
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
      
    Args:
      path: a file path (this can be local or on colossus)
      
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    MAX_SIZE = (1440, 1080)
    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(path)
    # print("Image size before: {}".format(image.size))
    image.thumbnail(MAX_SIZE)
    # print("Image size after resized: {}".format(image.size))
    (im_width, im_height) = image.size
    return np.array(image).astype(np.uint8)


def rms_contrast(image):
    return np.std(image)

def get_image_features(image_data_df, model_path):
    print("[*] Extracting image object features...")
    
    # Object detection model path
    # model_path = '../Image_Models/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8/saved_model/'
    threshold = 0.4

    # Use Pretrained model to get number of people, motorcycles, cars
    # Load the COCO Label Map
    category_index = {
        1: {'id': 1, 'name': 'person'},
        2: {'id': 2, 'name': 'bicycle'},
        3: {'id': 3, 'name': 'car'},
        4: {'id': 4, 'name': 'motorcycle'},
        5: {'id': 5, 'name': 'airplane'},
        6: {'id': 6, 'name': 'bus'},
        7: {'id': 7, 'name': 'train'},
        8: {'id': 8, 'name': 'truck'},
        9: {'id': 9, 'name': 'boat'},
        10: {'id': 10, 'name': 'traffic light'},
        11: {'id': 11, 'name': 'fire hydrant'},
        13: {'id': 13, 'name': 'stop sign'},
        14: {'id': 14, 'name': 'parking meter'},
        15: {'id': 15, 'name': 'bench'},
        16: {'id': 16, 'name': 'bird'},
        17: {'id': 17, 'name': 'cat'},
        18: {'id': 18, 'name': 'dog'},
        19: {'id': 19, 'name': 'horse'},
        20: {'id': 20, 'name': 'sheep'},
        21: {'id': 21, 'name': 'cow'},
        22: {'id': 22, 'name': 'elephant'},
        23: {'id': 23, 'name': 'bear'},
        24: {'id': 24, 'name': 'zebra'},
        25: {'id': 25, 'name': 'giraffe'},
        27: {'id': 27, 'name': 'backpack'},
        28: {'id': 28, 'name': 'umbrella'},
        31: {'id': 31, 'name': 'handbag'},
        32: {'id': 32, 'name': 'tie'},
        33: {'id': 33, 'name': 'suitcase'},
        34: {'id': 34, 'name': 'frisbee'},
        35: {'id': 35, 'name': 'skis'},
        36: {'id': 36, 'name': 'snowboard'},
        37: {'id': 37, 'name': 'sports ball'},
        38: {'id': 38, 'name': 'kite'},
        39: {'id': 39, 'name': 'baseball bat'},
        40: {'id': 40, 'name': 'baseball glove'},
        41: {'id': 41, 'name': 'skateboard'},
        42: {'id': 42, 'name': 'surfboard'},
        43: {'id': 43, 'name': 'tennis racket'},
        44: {'id': 44, 'name': 'bottle'},
        46: {'id': 46, 'name': 'wine glass'},
        47: {'id': 47, 'name': 'cup'},
        48: {'id': 48, 'name': 'fork'},
        49: {'id': 49, 'name': 'knife'},
        50: {'id': 50, 'name': 'spoon'},
        51: {'id': 51, 'name': 'bowl'},
        52: {'id': 52, 'name': 'banana'},
        53: {'id': 53, 'name': 'apple'},
        54: {'id': 54, 'name': 'sandwich'},
        55: {'id': 55, 'name': 'orange'},
        56: {'id': 56, 'name': 'broccoli'},
        57: {'id': 57, 'name': 'carrot'},
        58: {'id': 58, 'name': 'hot dog'},
        59: {'id': 59, 'name': 'pizza'},
        60: {'id': 60, 'name': 'donut'},
        61: {'id': 61, 'name': 'cake'},
        62: {'id': 62, 'name': 'chair'},
        63: {'id': 63, 'name': 'couch'},
        64: {'id': 64, 'name': 'potted plant'},
        65: {'id': 65, 'name': 'bed'},
        67: {'id': 67, 'name': 'dining table'},
        70: {'id': 70, 'name': 'toilet'},
        72: {'id': 72, 'name': 'tv'},
        73: {'id': 73, 'name': 'laptop'},
        74: {'id': 74, 'name': 'mouse'},
        75: {'id': 75, 'name': 'remote'},
        76: {'id': 76, 'name': 'keyboard'},
        77: {'id': 77, 'name': 'cell phone'},
        78: {'id': 78, 'name': 'microwave'},
        79: {'id': 79, 'name': 'oven'},
        80: {'id': 80, 'name': 'toaster'},
        81: {'id': 81, 'name': 'sink'},
        82: {'id': 82, 'name': 'refrigerator'},
        84: {'id': 84, 'name': 'book'},
        85: {'id': 85, 'name': 'clock'},
        86: {'id': 86, 'name': 'vase'},
        87: {'id': 87, 'name': 'scissors'},
        88: {'id': 88, 'name': 'teddy bear'},
        89: {'id': 89, 'name': 'hair drier'},
        90: {'id': 90, 'name': 'toothbrush'},
    }
    
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Loading object detection model took: ' + str(elapsed_time) + ' seconds')
    images_list = image_data_df.image_path.tolist()
    
    num_people_list = []
    num_motorcycles_list = []
    num_cars_list = []
    rms_contrast_list = []
    
    elapsed = []
    
    for indx, image_path in enumerate(tqdm(images_list)):        
        # print("\nDetecting objects in image {}".format(os.path.basename(image_path)))
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = np.expand_dims(image_np, 0)
        start_time = time.time()
        detections = detect_fn(input_tensor)
                
        final_scores = np.squeeze(detections['detection_scores'])
        detected_object_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        
        threshold_predictions = [category_index.get(value) for index, value in enumerate(detected_object_classes) if final_scores[index] > threshold]
        
        num_people = len(list(filter(lambda person: person['name'] == 'person', threshold_predictions)))
        num_motorcycles = len(list(filter(lambda motorcycles: motorcycles['name'] == 'motorcycle', threshold_predictions)))
        num_cars = len(list(filter(lambda motorcycles: motorcycles['name'] == 'car', threshold_predictions)))

        num_people_list.append(num_people)
        num_motorcycles_list.append(num_motorcycles)
        num_cars_list.append(num_cars)
        
        # print("Using threshold {} for elimination".format(threshold))
        # print("Number of people detected: {}".format(num_people))
        # print("Number of motorcycles detected: {}".format(num_motorcycles))
        # print("Number of cars detected: {}".format(num_cars))
        
        # print("Getting rms contrast...")
        pil_image = Image.fromarray(image_np)
        gray_image = pil_image.convert('L')
        gray_image_np = np.array(gray_image)
        rms = rms_contrast(gray_image_np)
        rms_contrast_list.append(rms)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed.append(elapsed_time)
        # print("Finished process image in {} seconds".format(elapsed_time))
    
    mean_elapsed = sum(elapsed) / float(len(elapsed))
    print("Total processed time: {}".format(sum(elapsed)))
    print("Process all images took:" + str(mean_elapsed) + ' second per image')
    
    # Add all the features to the dataframe
    frame = {'num_people': num_people_list, 'num_motorcycles': num_motorcycles_list, 'num_cars': num_cars_list, 'rms_contrast': rms_contrast_list}
    
    image_features = pd.DataFrame(frame)
    
    # labels = image_data_df[['pm25', 'aqi', 'aqi_rank']]
    
    # results = image_data_df[['image_path']]
    # results = pd.concat([results, image_features], axis=1)
    
    # print("Saving features to csv...")
    # results.to_csv(os.path.join(data_processed_dir, 'image_features_with_labels.csv'), index=True, sep=',', header=True)
    # print("Done")
    
    return image_features