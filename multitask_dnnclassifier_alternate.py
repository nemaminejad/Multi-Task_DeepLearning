
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import numpy as np
import os
import time 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

he_init = tf.variance_scaling_initializer()
scale = 0.001
### in the following I have not added max norm regularization and no learning rate scheduling
############################################################################################
############################################################################################

class MultiDNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden_layers=4, n_neurons=None, optimizer_class=tf.train.AdamOptimizer, learning_rate= 0.01, 
                 batch_size=199, activation=tf.nn.elu,initializer=he_init, batch_norm_momentum= None, 
          dropout_rate= 0.25,  w_max_n_thresh=None , weight_regularizer=None,
          random_state= 42, problem_scope= "dnn_1",learn_decay = None):
        
        # self._session = None
        self.n_hidden_layers =n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.w_max_n_thresh = w_max_n_thresh  #probably not use this one for now
        self.weight_regularizer = weight_regularizer #tf.contrib.layers.l1_regularizer(scale) with scale = 0.001
        self.problem_scope = problem_scope
        self.learn_decay = learn_decay
        self._session = None
        self.inefficient_params = False
        
    def _set_params(self, parameters):  
        self.n_hidden_layers = parameters["n_hidden_layers"]
        self.n_neurons = parameters["n_neurons"]
        self.optimizer_class = parameters["optimizer_class"]
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.activation = parameters["activation"]
        self.initializer = parameters["initializer"]
        self.batch_norm_momentum = parameters["batch_norm_momentum"]
        self.dropout_rate = parameters["dropout_rate"]
        self.random_state = parameters["random_state"]
        self.w_max_n_thresh = parameters["w_max_n_thresh"]  #probably not use this one for now
        self.weight_regularizer = parameters["weight_regularizer"]   #tf.contrib.layers.l1_regularizer(scale) with scale = 0.001
        self.learn_decay = parameters["learn_decay"]
        
    
    def _dnn(self, inputs):  # the construction of graph
        """Build the hidden layers, with support for batch normalization and dropout."""
        # with tf.container(self.problem_scope):   # to have a unique scope name when running multiple NN on one machine
        if type(self.n_neurons) is list:  # meaning that number of neurons changes for each layer
            for layer in range(self.n_hidden_layers):
                if self.dropout_rate:
                    inputs = tf.layers.dropout(inputs, self.dropout_rate, self._training)
                inputs = tf.layers.dense(inputs,self.n_neurons[layer], kernel_initializer=self.initializer,
                                         kernel_regularizer = self.weight_regularizer,name = "hidden%d" %(layer+1)) 
                if self.batch_norm_momentum:
                    inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,training=self._training)
                inputs = self.activation(inputs,name = "hidden%d_out"%(layer+1))
        else:
        
            for layer in range(self.n_hidden_layers):
                if self.dropout_rate:
                    inputs = tf.layers.dropout(inputs, self.dropout_rate, self._training)
                inputs = tf.layers.dense(inputs,self.n_neurons, kernel_initializer=self.initializer,
                                         kernel_regularizer = self.weight_regularizer,name = "hidden%d" %(layer+1)) 
                if self.batch_norm_momentum:
                    inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,training=self._training)
                inputs = self.activation(inputs,name = "hidden%d_out"%(layer+1))
                
        
            
        
        return inputs
        
    

    def _build_graph(self, n_inputs, n_outputs1, n_outputs2):
        # with tf.container(self.problem_scope):   # to have a unique scope name when running multiple NN on one machine
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        
        X = tf.placeholder(tf.float32, shape = (None, n_inputs),name = "X")
        y1 = tf.placeholder(tf.int32, shape = (None),name = "y1")
        y2 = tf.placeholder(tf.int32, shape = (None),name = "y2")
        
        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None
           # if the number of neurons for each layer will be different
        dnn_outputs= self._dnn(X)
        
#         logits = tf.layers.dense(dnn_out, n_outputs, kernel_initializer= self.initializer,
#                                  kernel_regularizer = self.weight_regularizer,name = "output" ) # no max norm for last layer
        logits1 = tf.layers.dense(dnn_outputs, n_outputs1, kernel_initializer=self.initializer, name="logits1")
        logits2 = tf.layers.dense(dnn_outputs, n_outputs2, kernel_initializer=self.initializer, name="logits2")
        
        Y_proba1 = tf.nn.softmax(logits1, name="Y_proba1")
        Y_proba2 = tf.nn.softmax(logits2, name="Y_proba2")
        
        ####### Loss and optimization:
        
        ### to take into account imbalanced issue:
        
        # # # # # xentropy = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=weight)
        # # # # weight_per_label = tf.placeholder(tf.float32, shape = (None),name = "sample_weight")
        # # # # xent = tf.multiply(weight_per_label,
                # # # # tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)) #shape [1, batch_size]
        # # # # xentropy = tf.reduce_mean(xent) #shape 1
        
        ### if not taking into account imbalanced issue:
        xentropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y1, logits = logits1)
        xentropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y2, logits = logits2)
        # calculate loss
        base_loss_1 = tf.reduce_mean(xentropy1, name="avg_xentropy1")
        base_loss_2 = tf.reduce_mean(xentropy2, name="avg_xentropy2")
        
        
        if self.weight_regularizer:    
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_1 = tf.add_n([base_loss_1] + reg_losses, name="loss1")
            loss_2 = tf.add_n([base_loss_2] + reg_losses, name="loss2")
        else: 
            loss_1 = base_loss_1
            loss_2 = base_loss_2
        
        if self.w_max_n_thresh:
            self.clip_all_weights = tf.get_collection("max_norm")
        else:
            self.clip_all_weights = None
        ## in case scheduling learnng rate was of interest
        # if self.optimizer_class.keywords['name'] !="Adam":
        if self.learn_decay is not None:
            initial_rate = self.learning_rate
            decay_steps = 10000
            decay_rate = 1/10
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(initial_rate, self.global_step, decay_steps, decay_rate)   
            optimizer = self.optimizer_class(learning_rate = self.learning_rate, global_step = self.global_step)
        else:
            self.global_step = None            
            #######  without learning rate decay:
            optimizer = self.optimizer_class(learning_rate = self.learning_rate)
            
            
        training_op_1 = optimizer.minimize(loss_1)
        training_op_2 = optimizer.minimize(loss_2)
        
        # extra ops for batch normalization:
        self._extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ######## performance
        
        correct1 = tf.nn.in_top_k(logits1, y1, 1)
        correct2 = tf.nn.in_top_k(logits2, y2, 1)
        
        accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32), name="accuracy1")
        accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32), name="accuracy2")
        
        #### initializer:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # Isolate the variables stored behind the scenes by the metric operation
#         running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_auc")

        # Define initializer to initialize/reset running variables
#         running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        running_vars_initializer= tf.local_variables_initializer()
        # Make the important operations available easily through instance variables
        self._X, self._y1, self._y2 = X, y1, y2
        self._Y_proba1,self._Y_proba2, self._loss_1, self._loss_2= Y_proba1, Y_proba2, loss_1, loss_2    
        self._training_op1, self._training_op2, self._accuracy1, self._accuracy2 = training_op_1,training_op_2, accuracy1, accuracy2
        self._init, self._saver = init, saver
        self._running_var = running_vars_initializer
        # self._weight_per_label = weight_per_label   # not needed if we are not doing class weight
        
#     def shuffle_batch(X, y, batch_size):
#         rnd_idx = np.random.permutation(len(X))
#         n_batches = len(X)//batch_size
#         for batch_idx in np.array_split(rnd_idx, n_batches):
#             X_batch, y_batch = X[batch_idx], y[batch_idx]
#             yield X_batch, y_batch
        
    def close_session(self):
        if self._session:
            self._session.close()
    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)
    
    def fit(self, X,y1, y2, n_epochs, X_valid = None, y_valid1 = None, y_valid2 = None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()
        
        n_inputs = X.shape[1]
        self.classes_1 = np.unique(y1)
        self.classes_2 = np.unique(y2)
        
        n_outputs1 = len(self.classes_1)
        n_outputs2 = len(self.classes_2)
        
        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1
        
        ### for y1
        self.class_to_index_1 = {label: index
                                for index, label in enumerate(self.classes_1)}
        y1 = np.array([self.class_to_index_1[label]
                      for label in y1], dtype=np.int32)
        ## for y2:              
        self.class_to_index_2 = {label: index
                                for index, label in enumerate(self.classes_2)}
        y2 = np.array([self.class_to_index_2[label]
                      for label in y2], dtype=np.int32)              
                      
        if y_valid1 is not None and  y_valid2 is not None:
            y_valid1 = np.array([self.class_to_index_1[label] for label in y_valid1], dtype=np.int32)
            y_valid2 = np.array([self.class_to_index_2[label] for label in y_valid2], dtype=np.int32)
        
        self._graph = tf.Graph()
        
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs1, n_outputs2)           
            
        
        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        
        ################### now train the model:
        self._session = tf.Session(graph = self._graph)
        with self._session.as_default() as sess:
            self._init.run()            
            # initialize/reset the running variables
            self._running_var.run()
            
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):                  
                    
                    X_batch, y_batch1, y_batch2 = X[rnd_indices], y1[rnd_indices], y2[rnd_indices]
                    
                    # sample_weight = compute_sample_weight(class_weight='balanced', y=y_batch)  #no weighting of samples here
                    feed_dict = {self._X: X_batch, self._y1: y_batch1, self._y2: y_batch2}#,self._weight_per_label: sample_weight}
                    
                    if self._training is not None:
                        feed_dict[self._training] = True
                    ### To implement alternate training:    
                    if np.random.rand() < 0.5:    
                        sess.run(self._training_op1, feed_dict=feed_dict)
                    else:
                        sess.run(self._training_op2, feed_dict=feed_dict)
                        
                        
                    if self._extra_update_ops:
                        sess.run(self._extra_update_ops, feed_dict=feed_dict)
                    if self.clip_all_weights:
                        sess.run(self.clip_all_weights)
                # for each epoch:
                if X_valid is not None and y_valid1 is not None and y_valid2 is not None:
                    
                    loss_val1, loss_val2, acc_val1, acc_val2 = sess.run([self._loss_1,self._loss_2, self._accuracy1, self._accuracy2],
                                                 feed_dict={self._X: X_valid, self._y1: y_valid1, self._y2: y_valid2}) #,self._weight_per_label: np.ones(y_valid.shape[0])})
                    
                    if loss_val1 < best_loss and loss_val2 < best_loss :
                       
                        best_loss = min(loss_val1, loss_val2 )
                        best_params = self._get_model_params()
                        checks_without_progress = 0
                    else:
                        checks_without_progress +=1
                    print("{}\tValidation loss1: {:.6f}\tloss2: {:.6f}\tBest loss: {:.6f}\tAccuracy1: {:.2f}\tAccuracy2: {:.2f}%".format(
                        epoch, loss_val1, loss_val2, best_loss, acc_val1 * 100, acc_val2 * 100))
                    
                    if checks_without_progress > max_checks_without_progress:
                        print("Early Stopping!")
                        break
                    if acc_val1 == 0 or acc_val2 ==0:
                        pritn("Bad parameter set")
                        self.inefficient_params = True
                        break
                else:
                    
                    loss_train1,loss_train2, acc_train1, acc_train2= sess.run([self._loss_1, self._loss_2, self._accuracy1, self._accuracy2],
                                                               feed_dict = {self._X:X_batch, self._y1:y_batch1 , self._y2:y_batch2})  #, self._weight_per_label: sample_weight})
                    print("{}\tLast training batch loss1: {:.6f}\tbatch loss2: {:.6f}\tAccuracy1: {:.2f}\tAccuracy2: {:.2f}%".format(
                        epoch, loss_train1,loss_train2, acc_train1 * 100, acc_train2 * 100))
            runtime = time.strftime('_%x_%X') 
            runtime = runtime.replace("/","-") 
            runtime = runtime.replace(":","-")
            save_model_path = os.path.join("//cluster/home/emaminej/scripts/DL/saved_nn/%s"%runtime)
            # save_path = self._saver.save(sess, os.path.join(save_model_path,"my_model_final.ckpt"))
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba1.eval(feed_dict={self._X: X}), self._Y_proba2.eval(feed_dict={self._X: X})
            
            
    def predict(self, X):
        y_score1, y_score2 = self.predict_proba(X)
        class_indices_1 = np.argmax(y_score1, axis=1)
        class_indices_2 = np.argmax(y_score2, axis=1)
        y_pred1 = np.array([[self.classes_1 [class_index]]
                         for class_index in class_indices_1], np.int32)
        y_pred2 = np.array([[self.classes_2 [class_index]]
                         for class_index in class_indices_2], np.int32)
        return y_pred1, y_pred2
        
        
    def _get_auc(self, X, y_true1,y_true2):   # returns auc for each task and the average auc
        
        n_classes1 = len(np.unique(y_true1))
        fpr1 = dict()
        tpr1 = dict()
        roc_auc1 = dict()
        y_true1 = label_binarize(y_true1, classes=list(np.unique(y_true1)))
        
        n_classes2 = len(np.unique(y_true2))
        fpr2 = dict()
        tpr2 = dict()
        roc_auc2 = dict()
        y_true2 = label_binarize(y_true2, classes=list(np.unique(y_true2)))
       
        y_score1, y_score2 = self.predict_proba(X)
        
        
        for i in range(n_classes1):
            fpr1[i], tpr1[i], _ = roc_curve(y_true1[:, i], y_score1[:, i])
            roc_auc1[i] = auc(fpr1[i], tpr1[i])
        # Compute micro-average ROC curve and ROC area
        fpr1["micro"], tpr1["micro"], _ = roc_curve(y_true1.ravel(), y_score1.ravel())
        roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])   
        
        for i in range(n_classes2):
            fpr2[i], tpr2[i], _ = roc_curve(y_true2[:, i], y_score2[:, i])
            roc_auc2[i] = auc(fpr2[i], tpr2[i])
        # Compute micro-average ROC curve and ROC area
        fpr2["micro"], tpr2["micro"], _ = roc_curve(y_true2.ravel(), y_score2.ravel())
        roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])    
        
        roc_auc_ave =  (roc_auc1["micro"] + roc_auc2["micro"])/2
            
        return  roc_auc1["micro"], roc_auc2["micro"], roc_auc_ave
        
    def save(self, path):
        self._saver.save(self._session, path)