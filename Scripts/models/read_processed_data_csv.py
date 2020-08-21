# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:34:00 2020

@author: dat18
"""

import pandas as pd
import os

# data_processed_dir = '../../../Data/Data HCM Processed/'

# random_split_dir = os.path.join(data_processed_dir, 'Combined Features + Global Weather', 'Random split')
# hist_split_dir = os.path.join(data_processed_dir, 'Combined Features + Global Weather', 'Historical split')

def get_train_test_data(split_dir, train_data_standardized_name, train_labels_name, test_data_standardized_name, test_labels_name):
    
    X_train = pd.read_csv(os.path.join(split_dir, train_data_standardized_name))
    y_train = pd.read_csv(os.path.join(split_dir, train_labels_name))
    
    X_test = pd.read_csv(os.path.join(split_dir, test_data_standardized_name))
    y_test = pd.read_csv(os.path.join(split_dir, test_labels_name))
   
    return X_train, y_train, X_test, y_test
    
    
# X_train_random_split, y_train_random_split, X_test_random_split, y_test_random_split = get_train_test_data_combined(random_split_dir)
# X_train_hist_split, y_train_hist_split, X_test_hist_split, y_test_hist_split = get_train_test_data_combined(hist_split_dir)

# idx = np.random.permutation(X_train_hist_split.index)
# #%%
# X_train_hist_split = X_train_hist_split.reindex(idx)
# y_train_hist_split = y_train_hist_split.reindex(idx)
# if __name__ == "__main__":
#     # X_train, y_train, X_validation, y_validation, X_test, y_test = get_train_val_test_data()
#      X_train, y_train, X_test, y_test = get_train_val_test_data()