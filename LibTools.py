from zlib import crc32
import os
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import sklearn
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, auc , hinge_loss
from sklearn.model_selection import StratifiedKFold 
from sklearn.base import clone
from scipy import interp
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
# PROJECT_ROOT_DIR = "data"


def save_fig(fig_name, root, tight_layout=True):
    ## saves the figure with given figure name in the root/plots/ directory
    path = os.path.join(root, "plots",  fig_name + ".png")
    print("Saving figure", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
#############################################################################   
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
#############################################################################    
# to split train and test set:
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def split_train_valid(train_set,test_ratio):
    np.random.seed(42)   # so that it always selects the same shuffled_indices
    shuffled_indices = np.random.permutation(len(train_set)) # to shuffle!!
    cross_valid_set_size = int(test_ratio * len(train_set))
    test_indices = shuffled_indices[:cross_valid_set_size]
    train_indices = shuffled_indices[cross_valid_set_size:]
    return train_set.iloc[train_indices], train_set.iloc[test_indices]    
#############################################################################

def CV_stratified(X,y,estimator,cv, method):
    #stratified CV of a model (estimator) to keep balanced number of positive/negative samples in different sets
    #estimator: the model object ( forexampl SVM model)
    #method : the method that the estimator uses to predict probabilities 
    # the method for SVM, is decision_function, and for RF or logistic regression it is predict_proba
    ### Example usage:
    ### estimator =  LinearSVC # a linear SVM model
    ### cv = 10  #for 10 fold cross validation
    ### method = predict_proba
    ###returns statistics within cross validations:  mean_valid_auc, std_valid_auc, mean_train_auc, std_train_auc
    ###
    skfolds = StratifiedKFold(n_splits = cv, random_state = 42)
    training_auc_list = []
    valid_auc_list = []
    for train_idx, valid_idx in skfolds.split(X, y):
        clone_clf = clone(estimator)
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]
        clone_clf.fit(X_train_fold, y_train_fold)
        
        if method =="decision_function": 
            y_vlaid_pred  = clone_clf.decision_function(X_valid_fold)
            y_train_pred = clone_clf.decision_function(X_train_fold)
            training_auc_list.append(roc_auc_score(y_train_fold,y_train_pred ))
            valid_auc_list.append(roc_auc_score(y_valid_fold,y_vlaid_pred ))
        if method == "predict_proba":
            y_vlaid_pred  = clone_clf.predict_proba(X_valid_fold)[:,1]
            y_train_pred = clone_clf.predict_proba(X_train_fold)[:,1]
            training_auc_list.append(roc_auc_score(y_train_fold,y_train_pred ))
            valid_auc_list.append(roc_auc_score(y_valdi_fold,y_valid_pred ))
          
    mean_valid_auc=np.mean(valid_auc_list)
    std_valid_auc=np.std(valid_auc_list)
    mean_train_auc=np.mean(training_auc_list)
    std_train_auc=np.std(training_auc_list)
    return mean_valid_auc, std_valid_auc, mean_train_auc, std_train_auc
    
  
#############################################################################


def indep_validation(X_set, y_set,  estimator, method):
    #validation of performance of the model on a  set when the input are Pandas data frames.
    ### outputs AUC for dataset
    ### sample usage: 
    ### estimator = LogisticRegression()
    ### method = "predict_proba
    ### X_set, y_set are Pandas data frame (pd.DataFrame object)
    ### indep_validation(XX_set, y_set, estimator, method)
    
    
    if method =="decision_function":  
        y_scores = estimator.decision_function(X_set)
        auc = roc_auc_score(y_set, y_scores)
    if method == "predict_proba":
        # the predict_proba returns n_samples * n_clsses array. [:,1] will be the positive class
        y_scores = estimator.predict_proba(X_set)[:,1]
        auc = roc_auc_score(y_set, y_scores)
        
    return auc

#############################################################################
def get_probabilities_all_data(dataset,estimator, method):
    # outputs the probability for the prediction of the class of all samples and the predicted class of each sample
    ### estimator = LogisticRegression()
    ### method = "predict_proba
    ### the dataset is Pandas data frame (pd.DataFrame object)
    ### get_probabilities_all_data(dataset,estimator, method)
    if method =="decision_function":    
        y_pred_prob = estimator.decision_function(dataset)        
    if method == "predict_proba":       
        y_pred_prob = estimator.predict_proba(dataset)[:,1]
    # the predicted class:    
    y_pred_label = estimator.predict(dataset)
    return  y_pred_prob, y_pred_label
    
 
#############################################################################    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlabel("Threshold") 
    plt.legend(loc="upper left") 
    plt.ylim([0, 1])

#############################################################################
def error_calculation (train_set, y_train, test_set, y_test):
    hinge_loss(unseen_y,pred)
#############################################################################  
def load_required_data(set1, set2, target, weight_needed):
    ## reads in CSVs of two input data and outputs Pandas dataframes of each set
    # class weight is calculated onl based on the data in the first csv
    set2 = pd.read_csv(set2,index_col="Name")
    set1 = pd.read_csv(set1,index_col="Name")
    
    # to remove unwanted index column:
    if "Unnamed: 0" in set2.columns:
        set2 = set2.drop(["Unnamed: 0"], axis = 1)
    if "Unnamed: 0" in set1.columns:
        set1 = set1.drop(["Unnamed: 0"], axis = 1)    
        
    ######################
    X1  = set1.drop(target,axis =1)
    y1 = set1[target]
    #########################

    X2 = set2.drop(target,axis = 1)
    y2 = set2[target]
    
    if weight_needed:   # only used for binary classes
        y_binary = (y1==1)
        weight_assign = X1.shape[0]/(2*np.bincount(y_binary))
        class_weight =  {0: weight_assign[0] , 1: weight_assign[1]}
    else:
        class_weight = None
    return X1, y1, class_weight, X2, y2
#############################################################################    
def load_finl_rand_eval(test_features,train_features, final_eval,target,feature_set,weight_needed ): 
    # loads data for final evaluation of the model on held out data from given CSVs. either when we want a set of selected descriptors or all the descriptors.
    #returns a large training by combining training set csv and the csv for validation set
    # It also returns a test set, which is teh held out set
    
    t_set = pd.read_csv(train_features,index_col="Name")
    v_set = pd.read_csv(test_features,index_col="Name")
    eval_set = pd.read_csv(final_eval,index_col="Name")
    
    ## to remove unwanted index column
    if "Unnamed: 0" in t_set.columns:
        t_set = t_set.drop(["Unnamed: 0"], axis = 1)
    if "Unnamed: 0" in v_set.columns:
        v_set = v_set.drop(["Unnamed: 0"], axis = 1)
    if "Unnamed: 0" in v_set.columns:
        eval_set = eval_set.drop(["Unnamed: 0"], axis = 1)
    

    all_training = pd.concat([t_set,v_set])
    X = all_training.drop(target,axis =1)
    y= all_training[target]
    
    unseen_X =eval_set.drop(target,axis =1)
    unseen_y = eval_set.loc[:,target]
    
    
    if weight_needed:
        weight_assign = X.shape[0]/(len(np.unique(y))*np.bincount(y))
        class_weight =  {0: weight_assign[0] , 1: weight_assign[1]}
    else:
        class_weight = None
        
    # to either select features or output all features
    if feature_set is not None:

        feature_path = os.path.join("data","feature_names", feature_set )
        features =pd.read_csv(feature_path)["featureNames"].tolist()
        new_features = []
        for f in features:
            if f in X.columns:
                new_features.append(f)
        X= X.loc[:,new_features]
        unseen_X = unseen_X.loc[:,new_features]

        
    return X, y, class_weight, unseen_X,unseen_y
    
 #############################################################################   
    
def load_temporal_data(valid_features, train_features, target, time_order_csv):
    ## reads the CSVs of input data, and load data into data frame by ordering the data based on time of compound development. 
    t_set = pd.read_csv(train_features,index_col="Name")
    v_set = pd.read_csv(valid_features,index_col="Name")
    
    if "Unnamed: 0" in t_set.columns:
        t_set = t_set.drop(["Unnamed: 0"], axis = 1) 
        v_set = v_set.drop(["Unnamed: 0"], axis = 1)    
    t_set = pd.concat([v_set,t_set])
    time = pd.read_csv(time_order_csv, index_col="VRT")
    
    time_p = time.loc[t_set.index.tolist()]
    training_temporal = pd.concat([time_p,t_set], axis =1)
    training_temporal = training_temporal.sort_values(by ="TemporalOrder", inplace = False)
    X = training_temporal.drop([target],axis =1)
    y = training_temporal[target]
    
    return X, y
    

    
#############################################################################
def class_report(y_true, y_pred, y_score=None, average='micro'):
    ###########  Usage for dnnclassifier:
        # # # dnn_clf = dnnclassifier() # a model
        # # # class_report(
        # # # y_true=unseen_y, 
        # # # y_pred=dnn_clf.predict(unseen_X)[:,0], 
        # # # y_score=dnn_clf.predict_proba(unseen_X))
    ######################################################
    
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df