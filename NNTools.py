import tensorflow as tf
import pandas as pd
from LibTools import class_report
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from dnnclassifier import DNNClassifier
from multitask_dnnclassifier_alternate import MultiDNNClassifier
from multitask_dnnclassifier_joined import MultiJoinedDNNClassifier


def leaky_relu (z, name=None):
    return tf.maximum(0.01 * z, z, name=name)
def selu(z,
         scale=1.05,
         alpha=1.67):
    return scale * elu(z, alpha)
    
    
def get_activation_fun(activation_fun):
    if activation_fun == 'tanh':
        return tf.nn.tanh
    elif activation_fun == 'relu6':
        return tf.nn.relu6
    elif activation_fun =='leaky_relu':
        return leaky_relu
    elif activation_fun == 'selu':
        return selu
    elif activation_fun == 'crelu':
        return tf.nn.crelu
    elif activation_fun == 'relu':
        return tf.nn.relu
    elif activation_fun == 'identity':
        return tf.identity
    elif activation_fun == 'linear':
        return None
    elif activation_fun == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception(
            "Invalid activation function {}".format(activation_fun))
            
            
def weight_var(shape, initilization):
    #shape : (n_inputs, n_neurons)
    stddev = 2/np.sqrt(n_inputs)
    weight_dist = tf.truncated_normal(shape, mean = 0, stddev =stddev)   (xavier initialization)
    w = tf.Variable(weight_dist,shape = shape, name = "kernel") # a 2D matrix of weights 
    
    # or say:
    if initialization == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.Variable(initializer=initializer, shape = shape,name = "kernel")
    return w
    
def bias_var(shape):
    #shape : (n_inputs, n_neurons)
    b = tf.Variable(tf.zeros[shape[1]], name = "bias")
    return b
	
	
def run_single_fit_search(X_train, y_train, unseen_X, unseen_y,param_distribs,target,category, n_iter, scoring, n_epochs):
	# run a single task Neural Network
    model = RandomizedSearchCV(DNNClassifier(), param_distribs, n_iter=n_iter,scoring = scoring,
                                    random_state=42, verbose=2, return_train_score=True)

    model.fit(X_train, y_train, X_valid=unseen_X, y_valid=unseen_y, n_epochs=n_epochs)
    output_dict = model.cv_results_ 
    df = pd.DataFrame(output_dict)
    runtime = time.strftime('_%x_%X') 
    runtime = runtime.replace("/","-") 
    runtime = runtime.replace(":","-")
    df.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis",target+"random_CV_dnn_results_%s.csv" %(runtime,category)))


    best_etimator_report = class_report(
        y_true=unseen_y, 
        y_pred=model.best_estimator_.predict(unseen_X)[:,0], 
        y_score=model.best_estimator_.predict_proba(unseen_X))
    best_etimator_report.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis",target+"sngl_random_CV_best_estimator_%s_%s.csv"%(runtime,category)))

# alternate training:    
def run_multi_fit_search(X_train,y_train_1,y_train_2,unseen_X, unseen_y_1, unseen_y_2,param_distribs, n_epochs,n_iter,name):
    gen_param = ParameterSampler(param_distribs, n_iter = n_iter, random_state=42)
    output_dict ={}
    p = 0
    frame = [pd.DataFrame()]

    for param_set in list(gen_param):
        p+=1  # to count number of parameters
        k=0
        
        model = MultiDNNClassifier()
        model._set_params(param_set)
        for fold_data in get_kfold(10,X_train, y_train_1, y_train_2): #y["var1"],y["var2"]):
            k+=1  # to count number of folds
            X_train_fold = fold_data[0]
            y_train1_fold = fold_data[1]
            y_train2_fold = fold_data[2]
            X_valid_fold = fold_data[3]  
            y_valid1_fold = fold_data[4]  
            y_valid2_fold = fold_data[5]   
        
            # use unseen_X and unsee_y_1 and unseen_y_2 for early stopping
            model.fit(X_train_fold, y_train1_fold,y2 = y_train2_fold, n_epochs=n_epochs, X_valid=unseen_X,y_valid1 =unseen_y_1, y_valid2 = unseen_y_2)
            output_dict.update({"auc_var1_train_%s" %str(k) : model._get_auc(X_train_fold,y_train1_fold,y_train2_fold)[0],
            "auc_var2_train_%s" %str(k) : model._get_auc(X_train_fold,y_train1_fold,y_train2_fold)[1],
            "auc_var1_valid_%s" %str(k) : model._get_auc(X_valid_fold,y_valid1_fold,y_valid2_fold)[0],
            "auc_var2_valid_%s" %str(k) : model._get_auc(X_valid_fold,y_valid1_fold,y_valid2_fold)[1]})
        if type(param_set['n_neurons']) is list:
            curr_list = param_set['n_neurons']
            param_set['n_neurons'] = "_".join([str(i) for i in curr_list])    
        output_dict.update(param_set)
        frame.append(pd.DataFrame(output_dict,index = [p]))
        
        
        runtime = time.strftime('_%x_%X') 
        runtime = runtime.replace("/","-") 
        runtime = runtime.replace(":","-")
        
        cwd = os.getcwd()
        outpath = os.path.join(cwd,"saved_nn",name)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        model.save(os.path.join(outpath,"model_%s_%s" %(str(p),runtime),"model_%s.ckpt" %str(p)))
         
        p+=1  # to count number of parameters
    df = pd.concat(frame)
   
    df.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis",name+runtime+".csv" ))
# joint training:
def run_multi_joined_fit_search(X_train,y_train_1,y_train_2,unseen_X, unseen_y_1, unseen_y_2,param_distribs, n_epochs,n_iter,name):
    gen_param = ParameterSampler(param_distribs, n_iter = n_iter, random_state=42)
    output_dict ={}
    p = 0
    frame = [pd.DataFrame()]

    for param_set in list(gen_param):
        p+=1  # to count number of parameters
        k=0
        
        model = MultiJoinedDNNClassifier()
        model._set_params(param_set)
        for fold_data in get_kfold(10,X_train, y_train_1, y_train_2): #y["var1"],y["var2"]):
            k+=1  # to count number of folds
            X_train_fold = fold_data[0]
            y_train1_fold = fold_data[1]
            y_train2_fold = fold_data[2]
            X_valid_fold = fold_data[3]  
            y_valid1_fold = fold_data[4]  
            y_valid2_fold = fold_data[5]   
        
            # use unseen_X and unsee_y_1 and unseen_y_2 for early stopping
            model.fit(X_train_fold, y_train1_fold,y2 = y_train2_fold, n_epochs=n_epochs, X_valid=unseen_X,y_valid1 =unseen_y_1, y_valid2 = unseen_y_2)
            if not model.inefficient_params:
                output_dict.update({"auc_var1_train_%s" %str(k) : model._get_auc(X_train_fold,y_train1_fold,y_train2_fold)[0],
                "auc_var2_train_%s" %str(k) : model._get_auc(X_train_fold,y_train1_fold,y_train2_fold)[1],
                "auc_var1_valid_%s" %str(k) : model._get_auc(X_valid_fold,y_valid1_fold,y_valid2_fold)[0],
                "auc_var2_valid_%s" %str(k) : model._get_auc(X_valid_fold,y_valid1_fold,y_valid2_fold)[1]})
        if type(param_set['n_neurons']) is list:
            curr_list = param_set['n_neurons']
            param_set['n_neurons'] = "_".join([str(i) for i in curr_list])    
        output_dict.update(param_set)
        frame.append(pd.DataFrame(output_dict,index = [p]))
        
        
        runtime = time.strftime('_%x_%X') 
        runtime = runtime.replace("/","-") 
        runtime = runtime.replace(":","-")
        
        cwd = os.getcwd()
        outpath = os.path.join(cwd,"saved_nn",name)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        model.save(os.path.join(outpath,"model_%s_%s" %(str(p),runtime),"model_%s.ckpt" %str(p)))
         
        
        p+=1  # to count number of parameters
    df = pd.concat(frame)


    df.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis",name+runtime+".csv" ))
    

def multi_task_eval(model, name, data,param_distribs,curr_path):
    X_train= data[0]
    y_train_1=data[1]
    y_train_2=data[2]
    n_epochs=data[3]
    unseen_X=data[4]
    unseen_y_1=data[5]
    unseen_y_2=data[6]
    
    model._set_params(param_distribs)
    model.fit(X_train,y_train_1, y_train_2, n_epochs, X_valid = unseen_X, y_valid1 = unseen_y_1, y_valid2 = unseen_y_2)

    # saving model and outputs:
    output_dict = {}
    frame = []
    if not model.inefficient_params:
        output_dict.update({"auc_var1_train" : model._get_auc(X_train,y_train_1, y_train_2)[0],
        "auc_var2_train" : model._get_auc(X_train,y_train_1, y_train_2)[1],
        "auc_var1_valid" : model._get_auc(unseen_X,unseen_y_1,unseen_y_2)[0],
        "auc_var2_valid" : model._get_auc(unseen_X,unseen_y_1,unseen_y_2)[1]})
        
        if type(param_distribs['n_neurons']) is list:
            curr_list = param_distribs['n_neurons']
            param_distribs['n_neurons'] = "_".join([str(i) for i in curr_list])    
            param_distribs['n_neurons'] = "_".join([str(i) for i in curr_list])    
        output_dict.update(param_distribs)
        frame.append(pd.DataFrame(output_dict,index = [1]))
        df = pd.concat(frame)
        runtime = time.strftime('_%x_%X') 
        runtime = runtime.replace("/","-") 
        runtime = runtime.replace(":","-")
        df.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis","final_evaluations",name+runtime+".csv" ))
        
        outpath = os.path.join(curr_path,"saved_nn",name)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        model.save(os.path.join(outpath,"model_%s" %runtime,"model_%s.ckpt" %name))
        
        
def sngl_task_eval(model, name, data,param_distribs,curr_path):
    X_train= data[0]
    y_train=data[1]   
    n_epochs=data[2]
    unseen_X=data[3]
    unseen_y=data[4]
    
    model._set_params(param_distribs)
    model.fit(X_train, y_train,n_epochs, X_valid = unseen_X, y_valid = unseen_y)
    output_dict = {}
    frame = []
    if not model.inefficient_params:
        output_dict.update({"auc_train" : get_auc_sngl_task(y_train,model.predict_proba(X_train),sample_weight=None),
                            "auc_valid": get_auc_sngl_task(unseen_y,model.predict_proba(unseen_X),sample_weight=None)})
    if type(param_distribs['n_neurons']) is list:
            curr_list = param_distribs['n_neurons']
            param_distribs['n_neurons'] = "_".join([str(i) for i in curr_list])    
            param_distribs['n_neurons'] = "_".join([str(i) for i in curr_list])    
    output_dict.update(param_distribs)
    frame.append(pd.DataFrame(output_dict,index = [1]))
    df = pd.concat(frame)
    runtime = time.strftime('_%x_%X') 
    runtime = runtime.replace("/","-") 
    runtime = runtime.replace(":","-")
    df.to_csv(os.path.join(PROJECT_ROOT_DIR, "analysis","final_evaluations",name+runtime+".csv" ))
    outpath = os.path.join(curr_path,"saved_nn",name)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    model.save(os.path.join(outpath,"model_%s" %runtime,"model_%s.ckpt" %name))
    
    