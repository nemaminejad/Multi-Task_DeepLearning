##################
# usage: an example of a script to run the prediction model that predicts two binary target variables simultaneously 
# Written by Nastaran Emaminejad
##################

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import time 
random_state =42
from LibTools import load_required_data
from functools import partial
from multitask_dnnclassifier_alternate import MultiDNNClassifier
from ref_DL_codes import run_multi_fit_search, max_norm_regularizer, leaky_relu



### get data

target = ["var1","var2"]
valid_features = os.path.join("data","multi_valid.csv")
train_features = os.path.join("data","multi_train.csv")

X, y, class_weight, unseen_X, unseen_y = load_required_data(valid_features, train_features, target,weight_needed = False)
n_f = X.shape[1]
X_train, y_train_1, y_train_2, class_weight, unseen_X, unseen_y_1, unseen_y_2 = X.values.astype(np.float32), y["var1"].values.astype(np.int32), y["var2"].values.astype(np.int32), class_weight, unseen_X.values.astype(np.float32), unseen_y["var1"].values.astype(np.int32),unseen_y["var2"].values.astype(np.int32)


# we want to identify the best model with respect to the validation set
# to do a search for parameters:
#So that in each iteration, randomly a set of parameters is chosen for training of the model
param_distribs = {
    "n_neurons": [100,300,500,1000,2000],
    "batch_size": [50,100,283,400],
    "learning_rate": [0.01, 0.02, 0.05],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    "n_hidden_layers": [0, 1, 2, 3, 4, 5],
    "optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer,name = 'Momentum', momentum=0.95),tf.train.GradientDescentOptimizer],
    "dropout_rate":[0.1,0.25,0.3,0.5],
    "weight_regularizer": [None, tf.contrib.layers.l1_regularizer(0.001)],
    #"w_max_n_thresh": [max_norm_regularizer(threshold = 1.0),],
    "batch_norm_momentum": [None, 0.9],
    "random_state" : [42,],
    "initializer": [tf.variance_scaling_initializer(),],
    "w_max_n_thresh":[None,],
    "learn_decay":[None,]
    
}

#number of iteration (or number of different combination of parameters)
# so that for example you will have 40 different models being trained on the training set
# details of the performance on the validation set for each of the models will be written in the resulting CSV file
n_epochs = 1000
n_iter = 40

name = "Multitask_alternate_dnn_results_"

run_multi_fit_search(X_train,y_train_1,y_train_2,unseen_X, unseen_y_1, unseen_y_2,param_distribs, n_epochs,n_iter,name)