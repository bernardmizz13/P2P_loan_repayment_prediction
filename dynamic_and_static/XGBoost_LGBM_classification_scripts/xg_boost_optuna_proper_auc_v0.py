import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
# for evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
# for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for confusion matrix
from sklearn.metrics import confusion_matrix as cf
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib
import numpy as np

#BM
NUM_PARALLEL_EXEC_UNITS = 6
#################################################################################################

# for file name saving
from datetime import datetime
# get date time string
datestring = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
# for time tracking
import time
# for paths and logging
import os
# for logging purposes
import logging

__file__ = './logs/xg_boost_log_' + str(datestring)

# initializing the logger
logging.basicConfig(filename=os.path.realpath(__file__) + '.log',
                    filemode='a+',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging.info("Started the script!")

pred_threshold = 0

logging.info("Probability threshold is " + str(pred_threshold))
print("Probability threshold is", str(pred_threshold))

#################################################################################################

# function for auc roc evaluation metric
def my_auc(preds, y):
    
    # return the score to be maximized
    fpr, tpr, thresholds = roc_curve(y, preds)
    res = auc(fpr, tpr)

    # perform kappa calculation
    return 'my_auc', res

#################################################################################################

# these are the categorical features
features_to_remove = [
'grade_A',
'grade_B',
'grade_C',
'grade_D',
'grade_E',
'grade_F',
'sub_grade_A1',
'sub_grade_A2',
'sub_grade_A3',
'sub_grade_A4',
'sub_grade_A5',
'sub_grade_B1',
'sub_grade_B2',
'sub_grade_B3',
'sub_grade_B4',
'sub_grade_B5',
'sub_grade_C1',
'sub_grade_C2',
'sub_grade_C3',
'sub_grade_C4',
'sub_grade_C5',
'sub_grade_D1',
'sub_grade_D2',
'sub_grade_D3',
'sub_grade_D4',
'sub_grade_D5',
'sub_grade_E1',
'sub_grade_E2',
'sub_grade_E3',
'sub_grade_E4',
'sub_grade_E5',
'sub_grade_F1',
'sub_grade_F2',
'sub_grade_F3',
'sub_grade_F4',
'sub_grade_F5',
'sub_grade_G1',
'sub_grade_G2',
'sub_grade_G3',
'sub_grade_G4',
'emp_length_1 year',
'emp_length_10+ years',
'emp_length_2 years',
'emp_length_3 years',
'emp_length_4 years',
'emp_length_5 years',
'emp_length_6 years',
'emp_length_7 years',
'emp_length_8 years',
'emp_length_9 years',
'home_ownership_ANY',
'home_ownership_MORTGAGE',
'home_ownership_NONE',
'home_ownership_OTHER',
'home_ownership_OWN',
'verification_status_Not Verified',
'verification_status_Source Verified',
'addr_state_AK',
'addr_state_AL',
'addr_state_AR',
'addr_state_AZ',
'addr_state_CA',
'addr_state_CO',
'addr_state_CT',
'addr_state_DC',
'addr_state_DE',
'addr_state_FL',
'addr_state_GA',
'addr_state_HI',
'addr_state_IA',
'addr_state_ID',
'addr_state_IL',
'addr_state_IN',
'addr_state_KS',
'addr_state_KY',
'addr_state_LA',
'addr_state_MA',
'addr_state_MD',
'addr_state_ME',
'addr_state_MI',
'addr_state_MN',
'addr_state_MO',
'addr_state_MS',
'addr_state_MT',
'addr_state_NC',
'addr_state_ND',
'addr_state_NE',
'addr_state_NH',
'addr_state_NJ',
'addr_state_NM',
'addr_state_NV',
'addr_state_NY',
'addr_state_OH',
'addr_state_OK',
'addr_state_OR',
'addr_state_PA',
'addr_state_RI',
'addr_state_SC',
'addr_state_SD',
'addr_state_TN',
'addr_state_TX',
'addr_state_UT',
'addr_state_VA',
'addr_state_VT',
'addr_state_WA',
'addr_state_WI',
'addr_state_WV',
'initial_list_status_f',
'application_type_DIRECT_PAY',
'application_type_INDIVIDUAL'
# 'earliest_cr_line_month',
# 'earliest_cr_line_year'
]

#################################################################################################
# function that computes my auc roc
def compute_my_auc_roc(orig_labels, predict_probs, optim, n):
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (16,9)
    
    score = predict_probs
    y = orig_labels
    
    # false positive rate
    FPR = []
    # true positive rate
    TPR = []
    # Iterate thresholds from 0.0 to 1.0, 20 in total
    thresholds = np.arange(0.0, 1.01, 0.05)
    
    # get number of positive and negative examples in the dataset
    P = sum(y)
    N = len(y) - P
    
    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP=0
        TP=0
        thresh = round(thresh,2) #Limiting floats to two decimal points, or threshold 0.6 will be 0.6000000000000001 which gives FP=0
        for i in range(len(score)):
            if (score[i] >= thresh):
                if y[i] == 1:
                    TP = TP + 1
                if y[i] == 0:
                    FP = FP + 1
        FPR.append(FP/N)
        TPR.append(TP/P)
    
    # This is the AUC
    #you're integrating from right to left. This flips the sign of the result
    auc = -1 * np.trapz(TPR, FPR)
    
    if optim is False:
    
        plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve, AUC = %.2f'%auc)
        plt.legend(loc="lower right")
        
        plt.savefig('./my_auc_roc_curves/model_' + str(n) + '_my_auc_roc_prob.png')
        plt.close()
    
    return auc

# function to save metrics
def save_metrics(iteration, num, trial_number, orig_labels, raw_predictions, label_predictions):
    
    # auc roc
    
    # get the fpr, tpr and threshold
    fpr_raw, tpr_raw, thresholds_raw = roc_curve(orig_labels, raw_predictions)
    auc_raw = round(auc(fpr_raw, tpr_raw), 6)
    
    fpr_label, tpr_label, thresholds_label = roc_curve(orig_labels, label_predictions)
    auc_label = round(auc(fpr_label, tpr_label), 6)
    
    ###############################################
    
    accuracy = accuracy_score(orig_labels, label_predictions)
    precision = precision_score(orig_labels, label_predictions)
    recall = recall_score(orig_labels, label_predictions)
    f1 = f1_score(orig_labels, label_predictions)
    
    logging.info("Trial number: " + str(trial_number))
    logging.info("Iteration " + iteration)
    logging.info("Model number: " + str(num))
    logging.info("auc roc label: " + str(auc_label))
    logging.info("auc roc raw: " + str(auc_raw))
    logging.info("accuracy " + str(accuracy))
    logging.info("precision " + str(precision))
    logging.info("recall " + str(recall))
    logging.info("f1 " + str(f1))
    
    # dataframe for trail results
    results_df = pd.DataFrame(columns = ['model_num', 'trial_no', 'iteration', 'auc_roc_label', 'auc_roc_raw', 'my_auc_roc','accuracy', 'precision', 'recall', 'f1_score'])
    
    # path to save the trial metrics
    filepath = 'trial_metrics/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # read the trial number
    f = open("trial_no.txt", "r")
    trial_no = f.read()
    trial_no = int(trial_no)
    f.close()
    
    # if first trial create csv
    if os.path.exists(filepath + 'model_' + str(num) + '_metrics.csv') is False:
        results_df.to_csv(filepath + 'model_' + str(num) + '_metrics.csv', index = False)
    else:
        results_df = pd.read_csv(filepath + 'model_' + str(num) + '_metrics.csv')
        
    our_auc_roc = compute_my_auc_roc(orig_labels, raw_predictions, True, num)
    
    # add the new metrics
    results_df = results_df.append(pd.DataFrame({'model_num': [num], 'trial_no': [trial_number], 'iteration': [iteration], 'auc_roc_label': [auc_label], 'auc_roc_raw': [auc_raw], 'my_auc_roc': [our_auc_roc], 'accuracy': [accuracy],
                                                'precision': [precision], 'recall': [recall], 'f1_score': [f1]}))
    # save to csv
    results_df.to_csv(filepath + 'model_' + str(num) + '_metrics.csv', index = False)
    
    ###############################################
    
# function that prepares the validation data
def prepare_validation_data(val_random, validation_set_name, validation_set, full_train, remove_features):
    
    Y_val = pd.DataFrame()
    X_val = pd.DataFrame()
    
    Y_train = pd.DataFrame()
    X_train = pd.DataFrame()
    
    # if there was no validation set passed
    if len(validation_set_name) == 0:
    
        if val_random is True:
            # select random validation set
            # split train in validation and train            
            val_set = full_train.sample(frac = 0.15, random_state=22903)             
            train_set = full_train.drop(val_set.index)
            
            # store the validation sets
            Y_val = val_set.loan_status
            X_val = val_set.drop(['loan_status'], axis = 1)
            
            # store the train sets
            Y_train = train_set.loan_status
            X_train = train_set.drop(['loan_status'], axis = 1)
            
            logging.info('Validation set chosen randomly!')
        else:
            # select last percentage
            # split train in validation and train 
            val_set = full_train.tail(int(len(full_train.index)*(0.15)))
            train_set = full_train.drop(val_set.index)            
            
            # store the validation sets
            Y_val = val_set.loan_status
            X_val = val_set.drop(['loan_status'], axis = 1)
            
            # store the train sets
            Y_train = train_set.loan_status
            X_train = train_set.drop(['loan_status'], axis = 1)
            
            logging.info('Validation set not chosen randomly!')
            
    else:
        # read the passed validation set
        val_set = pd.read_csv(validation_set, sep = ',')
        
        Y_val = val_set.loan_status
        X_val = val_set.drop(['loan_status'], axis = 1)
        
        # drop features if asked to
        if remove_features is True:
            X_val = X_val.drop(features_to_remove, axis = 1)
            logging.info('Categorical features removed from validation set!')
        else:
            logging.info('Categorical features not removed from validation set!')
        
        Y_train = full_train.loan_status
        X_train = full_train.drop(['loan_status'], axis = 1)
    
    # returned the prepared sets
    return X_train, Y_train, X_val, Y_val
    
    #########################################
    
# function that prepares the scaling of the data
def prepare_data_scaling(scaled, X_train, X_val, X_test):
    # check if we require scaling the data
    if scaled is True:
        # Standardize features by removing the mean and scaling to unit variance, mean of 0 and standard deviation of 1
        # Scale values of numerical columns
        df_columns = X_train.columns.tolist()
        
        # create a scaler
        scaler = StandardScaler(copy=False)
        
        # store also the columns which require scaling
        columns_to_scale = list(set(df_columns) - set(features_to_remove))
        
        # scale
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
        
        # X_train.to_csv('train_set_scaled.csv', sep = ',', index = False)
        # X_val.to_csv('validation_set_scaled.csv', sep = ',', index = False)
        # X_test.to_csv('test_set_scaled.csv', sep = ',', index = False)
        # 
        # # re-read for windows format purposes
        # X_train = pd.read_csv("train_set_scaled.csv", sep = ',')
        # X_val = pd.read_csv("validation_set_scaled.csv", sep = ',')
        # X_test = pd.read_csv("test_set_scaled.csv", sep = ',')
        
        logging.info('Data scaled and saved for review!')
        
        # return the scaled data
        return X_train, X_val, X_test
    else:
        logging.info('Data not scaled!')
        
        # return the original untouched data
        return X_train, X_val, X_test

# function that resamples the data using a sampling technique
def resample_the_data(seed_no, sampling_technique, X_train, Y_train):
    
    if sampling_technique == 'RU':
        rus = RandomUnderSampler(random_state=seed_no)
        train_x_not_sampled = X_train
        train_y_not_sampled = Y_train
        X_train, Y_train = rus.fit_sample(train_x_not_sampled, train_y_not_sampled)
        logging.info("Sampled using the random undersampler")
        logging.info(X_train.shape)
        logging.info(Y_train.shape)
        unique, counts = np.unique(Y_train, return_counts=True)
        logging.info((dict(zip(unique, counts))))
    if sampling_technique == 'SM':
        sm = SMOTE(random_state=seed_no)
        train_x_not_sampled = X_train
        train_y_not_sampled = Y_train
        X_train, Y_train = sm.fit_sample(train_x_not_sampled, train_y_not_sampled)
        logging.info("Sampled using the regular smote")
        logging.info(X_train.shape)
        logging.info(Y_train.shape)
        unique, counts = np.unique(Y_train, return_counts=True)
        logging.info((dict(zip(unique, counts))))
    if sampling_technique == 'SM_BL_1':
        sm = SMOTE(kind='borderline1', random_state=seed_no)
        train_x_not_sampled = X_train
        train_y_not_sampled = Y_train
        X_train, Y_train = sm.fit_sample(train_x_not_sampled, train_y_not_sampled)
        logging.info("Sampled using the smote borderline 1")
        logging.info(X_train.shape)
        logging.info(Y_train.shape)
        unique, counts = np.unique(Y_train, return_counts=True)
        logging.info((dict(zip(unique, counts))))
    if sampling_technique == 'SM_BL_2':
        sm = SMOTE(kind='borderline2', random_state=seed_no)
        train_x_not_sampled = X_train
        train_y_not_sampled = Y_train
        X_train, Y_train = sm.fit_sample(train_x_not_sampled, train_y_not_sampled)
        logging.info("Sampled using the smote borderline 2")
        logging.info(X_train.shape)
        logging.info(Y_train.shape)
        unique, counts = np.unique(Y_train, return_counts=True)
        logging.info((dict(zip(unique, counts))))
    if sampling_technique == 'RO':
        sm = RandomOverSampler(random_state=seed_no)
        train_x_not_sampled = X_train
        train_y_not_sampled = Y_train
        X_train, Y_train = sm.fit_sample(train_x_not_sampled, train_y_not_sampled)
        logging.info("Sampled using the random oversampler")
        logging.info(X_train.shape)
        logging.info(Y_train.shape)
        unique, counts = np.unique(Y_train, return_counts=True)
        logging.info((dict(zip(unique, counts))))
        
    return X_train, Y_train

# function that prints information about our data
def print_data_info(X_train, Y_train, X_val, Y_val, X_test, Y_test):

    logging.info("\nOutside trial")
    
    logging.info("\ntrain shape:")
    logging.info(str(X_train.shape))
    logging.info(str(Y_train.shape))
    
    logging.info("\nvalidation shape:")
    logging.info(str(X_val.shape))
    logging.info(str(Y_val.shape))
    
    logging.info("\ntest shape:")
    logging.info(str(X_test.shape))
    logging.info(str(Y_test.shape))
    
    print("\ntrain shape:")
    print(str(X_train.shape))
    print(str(Y_train.shape))
    
    print("\nvalidation shape:")
    print(str(X_val.shape))
    print(str(Y_val.shape))
    
    print("\ntest shape:")
    print(str(X_test.shape))
    print(str(Y_test.shape))

# save the best trial metrics to a csv  
def save_best_trial_metrics(parameters, num, val_model, val_orig, val_preds, val_raw, test_model, test_orig, test_preds, test_raw, my_auc_val, my_auc_test, sklearn_test_auc_roc, sklearn_val_auc_roc):
    
    # folder for confution matrices and predictions
    filepath = 'final_model_confusion_matrices/model_'+str(num)+'/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # produce the confusion matrix for test
    conf_mat = cf(test_orig, list(test_preds))
    test_tn = conf_mat[0][0]
    test_fp = conf_mat[0][1]
    test_fn = conf_mat[1][0]
    test_tp = conf_mat[1][1]
    
    name = filepath + "model_" + str(num) + '_test_sklearn_confusion_matrix.txt'
    f = open(name, 'w')
    f.write('\n\nConfusion Matrix\n\n{}\n'.format(conf_mat))
    f.close()
        
    # produce the confusion matrix for test
    conf_mat = cf(val_orig, list(val_preds))
    val_tn = conf_mat[0][0]
    val_fp = conf_mat[0][1]
    val_fn = conf_mat[1][0]
    val_tp = conf_mat[1][1]
    
    name = filepath + "model_" + str(num) + '_validation_sklearn_confusion_matrix.txt'
    f = open(name, 'w')
    f.write('\n\nConfusion Matrix\n\n{}\n'.format(conf_mat))
    f.close()
    
    # save the images
    save_auc_roc_curves(num, val_orig, val_preds, val_raw, test_orig, test_preds, test_raw)
    
    #####################################################
    
    # path to save the trial metrics
    filepath = 'final_model_metrics/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    results_df = pd.DataFrame(columns = ['model_num', 'test_auc_roc_binary', 'test_auc_roc_prob', 'my_test_auc_roc', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_tp', 'test_fp', 'test_tn', 'test_fn', 'parameters',
    'val_auc_roc_binary', 'val_auc_roc_prob', 'my_val_auc_roc', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_tp', 'val_fp', 'val_tn', 'val_fn'])
    
    # if first model
    if os.path.exists(filepath + 'final_model_metrics.csv') is False:
        results_df.to_csv(filepath + 'final_model_metrics.csv', index = False)
    else:
        results_df = pd.read_csv(filepath + 'final_model_metrics.csv')
        
    test_accuracy = accuracy_score(test_orig, test_preds)
    test_precision = precision_score(test_orig, test_preds)
    test_recall = recall_score(test_orig, test_preds)
    test_f1 = f1_score(test_orig, test_preds)
    
    val_accuracy = accuracy_score(val_orig, val_preds)
    val_precision = precision_score(val_orig, val_preds)
    val_recall = recall_score(val_orig, val_preds)
    val_f1 = f1_score(val_orig, val_preds)
        
    # add the new metrics
    results_df = results_df.append(pd.DataFrame({'model_num': [num], 'test_auc_roc_binary': [test_model], 'test_auc_roc_prob': sklearn_test_auc_roc, 'my_test_auc_roc': [my_auc_test], 'test_accuracy': [test_accuracy], 'test_precision': [test_precision], 'test_recall': [test_recall],
    'test_f1': [test_f1], 'test_tp': [test_tp], 'test_fp': [test_fp], 'test_tn': [test_tn], 'test_fn': [test_fn], 'parameters': [parameters], 'val_auc_roc_binary': [val_model], 'val_auc_roc_prob': sklearn_val_auc_roc, 'my_val_auc_roc': [my_auc_val], 'val_accuracy': [val_accuracy], 'val_precision': [val_precision], 'val_recall': [val_recall], 'val_f1': [val_f1],
    'val_tp': [val_tp], 'val_fp': [val_fp], 'val_tn': [val_tn], 'val_fn': [val_fn]}))
    
    # save to csv
    results_df.to_csv(filepath + 'final_model_metrics.csv', index = False)

# function that saves auc roc
def save_auc_roc_curves(num, val_orig, val_preds, val_raw, test_orig, test_preds, test_raw):

    # folder for confution matrices and predictions
    filepath = 'best_trial_auc_roc_curves/model_'+str(num)+'/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    fpr, tpr, _ = [], [], []
    fpr_prob, tpr_prob, _prob = [], [], []
    
    # get auc of roc with the binary labels
    fpr, tpr, _ = roc_curve(val_orig, val_preds)
    fpr_prob, tpr_prob, _prob = roc_curve(val_orig, val_raw)
        
    # BM plotting AUC ROC
    plt.figure()
    plt.plot(fpr, tpr, '--r') # dashed red
    plt.plot([0, 1], [0, 1], '-b') # straigt line blue
    plt.title('AUC ROC')
    # close the figure
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig(filepath + 'model_' + str(num) + '_validation_binary.png')
    plt.close()
    
    # BM plotting AUC ROC
    plt.figure()
    plt.plot(fpr_prob, tpr_prob, '--r') # dashed red
    plt.plot([0, 1], [0, 1], '-b') # straigt line blue
    plt.title('AUC ROC')
    # close the figure
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig(filepath + 'model_' + str(num) + '_validation_prob.png')
    plt.close()
    
    fpr, tpr, _ = [], [], []
    fpr_prob, tpr_prob, _prob = [], [], []
    
    # get auc of roc with the binary labels
    fpr, tpr, _ = roc_curve(test_orig, test_preds)
    fpr_prob, tpr_prob, _prob = roc_curve(test_orig, test_raw)
        
    # BM plotting AUC ROC
    plt.figure()
    plt.plot(fpr, tpr, '--r') # dashed red
    plt.plot([0, 1], [0, 1], '-b') # straigt line blue
    plt.title('AUC ROC')
    # close the figure
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig(filepath + 'model_' + str(num) + '_test_binary.png')
    plt.close()
    
    # BM plotting AUC ROC
    plt.figure()
    plt.plot(fpr_prob, tpr_prob, '--r') # dashed red
    plt.plot([0, 1], [0, 1], '-b') # straigt line blue
    plt.title('AUC ROC')
    # close the figure
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig(filepath + 'model_' + str(num) + '_test_prob.png')
    plt.close()



# callback function to store the best xgboost
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

#################################################################################################

# BM read the training and testing set names
hyperparameters = pd.read_csv('./hyperparameters/hyperparameters.csv', sep = ',')

# Formatted the the parameters
hyperparameters.objective =  hyperparameters.objective.apply(lambda x: x[1:-1].split(' '))
hyperparameters.scale_pos_weight =  hyperparameters.scale_pos_weight.apply(lambda x: x[1:-1].split(' '))
hyperparameters.learning_rate =  hyperparameters.learning_rate.apply(lambda x: x[1:-1].split(' '))
hyperparameters.gamma =  hyperparameters.gamma.apply(lambda x: x[1:-1].split(' '))
hyperparameters.max_depth =  hyperparameters.max_depth.apply(lambda x: x[1:-1].split(' '))
hyperparameters.min_child_weight =  hyperparameters.min_child_weight.apply(lambda x: x[1:-1].split(' '))
hyperparameters.subsample =  hyperparameters.subsample.apply(lambda x: x[1:-1].split(' '))
hyperparameters.colsample_bytree =  hyperparameters.colsample_bytree.apply(lambda x: x[1:-1].split(' '))
hyperparameters.reg_alpha =  hyperparameters.reg_alpha.apply(lambda x: x[1:-1].split(' '))
hyperparameters.reg_lambda =  hyperparameters.reg_lambda.apply(lambda x: x[1:-1].split(' '))
hyperparameters.colsample_bylevel = hyperparameters.colsample_bylevel.apply(lambda x: x[1:-1].split(' '))
hyperparameters.colsample_bynode = hyperparameters.colsample_bynode.apply(lambda x: x[1:-1].split(' '))

print("Starting for loop")
logging.info("Starting for loop")

# BM for each training and testing set
for index, row in hyperparameters.iterrows():

    pred_threshold = float(row['prediction_threshold'])
    
    # read the trial number
    f = open("trial_no.txt", "w")
    f.write("0")
    f.close()
    
    model_num = index
    
    # if set to run
    if row['run'] is True:
        
        # get the number of trials
        trials = int(row['trials'])
        
        # get the general settings for the xg boost
        val_random = row['val_random']
        remove_features = row['remove_features']
        scaled = row['scaled']
        apply_pca = row['apply_pca']
        
        sampling_technique = row['sampling_technique']
        
        print("Captured the general settings for the XG boost")
        logging.info("Captured the general settings for the XG boost")
        
        # get the train and test set names
        train_set_name = row['train_set_name']
        test_set_name = row['test_set_name']
        validation_set_name = row['validation_set_name']
        
        print("Captured the set names")
        logging.info("Captured the set names")
        
        # get the names with the path
        train_set = './data/' + row['train_set_name'] + '.csv'
        test_set = './data/' + row['test_set_name'] + '.csv'
        validation_set = './data/' + row['validation_set_name'] + '.csv'
        
        # load train dataset
        dataframe = pd.read_csv(train_set)
        dataframe.loan_status = dataframe.loan_status.astype(int)
        # dataframe.to_csv(train_set, sep = ',', index = False)
        # dataframe = pd.read_csv(train_set, sep = ',')
        
        # split into input (X) and output (Y) variables
        Y_train = dataframe.loan_status
        X_train = dataframe.drop(['loan_status'], axis = 1)
    
        # get the test data
        dataframe = pd.read_csv(test_set)
        dataframe.loan_status = dataframe.loan_status.astype(int)
        # dataframe.to_csv(test_set, sep = ',', index = False)
        # dataframe = pd.read_csv(test_set, sep = ',')
        
        # split into input (X) and output (Y) variables
        Y_test = dataframe.loan_status
        X_test = dataframe.drop(['loan_status'], axis = 1)
    
        print("Model no.", str(index))
        logging.info("Model no. " + str(index))
    
        print('Train and test data read!')
        logging.info('Train and test data read!')
        
        # drop features if asked to
        if remove_features is True:
            X_train = X_train.drop(features_to_remove, axis = 1)
            X_test = X_test.drop(features_to_remove, axis = 1)
            logging.info('Categorical features removed!')
        else:
            logging.info('Categorical features not removed!')
        
        # backup
        orig_X_train = X_train
        orig_Y_train = Y_train
        
        # combine train with the label
        full_train = X_train
        full_train['loan_status'] = Y_train
        
        # prepare the validation data
        X_train, Y_train, X_val, Y_val = prepare_validation_data(val_random, validation_set_name, validation_set, full_train, remove_features)
        
        # prepare the data for scaling
        X_train, X_val, X_test = prepare_data_scaling(scaled, X_train, X_val, X_test)
        
        # print the shape of the data
        print_data_info(X_train, Y_train, X_val, Y_val, X_test, Y_test)
        
        X_train = X_train.values
        Y_train = Y_train.values
        
        X_val = X_val.values
        Y_val = Y_val.values
        
        X_test = X_test.values
        Y_test = Y_test.values
        
        # function for resampling the data
        X_train, Y_train = resample_the_data(int(row['sampling_technique_seed']), sampling_technique, X_train, Y_train)
        
        # check whether to apply PCA
        if apply_pca is True:
            # import the PCA from sklearn
            from sklearn.decomposition import PCA
            # create the PCA instance
            pca = PCA(n_components=len(X_train[0]))
            X_train_PCA = pca.fit_transform(X_train)
            
            # getting the cumulative variance of the components
            var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
            
            # for loop the get the number of components with 99% variance
            print(len(var))
            print(var)
            print(pca.explained_variance_ratio_)
            n_of_components = 0
            reached = True
            while var[n_of_components] < 99:
                n_of_components += 1
                if n_of_components == 133:
                    reached = False
                    break
            
            if reached is False:
                n_of_components = 0
                while var[n_of_components] < 98:
                    n_of_components += 1
            
            # new pca for the components
            pca = PCA(n_components=n_of_components)
            
            # apply the PCA on the train, test and validation sets separately
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            
            logging.info("Number of columns after PCA " + str(len(X_train[0])))
            print("Number of columns after PCA", str(len(X_train[0])))
        
        # define function for the trial, will be called in each study
        def objective(trial):
            
            # read the trial number
            f = open("trial_no.txt", "r")
            trial_no = f.read()
            f.close()
            
            trial_no = int(trial_no)
            trial_no += 1
            
            f = open("trial_no.txt", "w")
            f.write(str(trial_no))
            f.close()
            
            scale = hyperparameters.scale_pos_weight[index]
            learningrate = hyperparameters.learning_rate[index]
            gamma = hyperparameters.gamma[index]
            maxdepth = hyperparameters.max_depth[index]
            minchildweight = hyperparameters.min_child_weight[index]
            subsample = hyperparameters.subsample[index]
            colsample = hyperparameters.colsample_bytree[index]
            regalpha = hyperparameters.reg_alpha[index]
            reg_lambda = hyperparameters.reg_lambda[index]
            colsample_bylevel = hyperparameters.colsample_bylevel[index]
            colsample_bynode = hyperparameters.colsample_bynode[index]
            
            # convert hyperparameters to float to be passed to the xg boost
            scale = [float(i) for i in scale]
            learningrate = [float(i) for i in learningrate]
            gamma = [float(i) for i in gamma]
            maxdepth = [int(i) for i in maxdepth]
            minchildweight = [float(i) for i in minchildweight]
            subsample = [float(i) for i in subsample]
            colsample = [float(i) for i in colsample]
            regalpha = [float(i) for i in regalpha]
            reg_lambda = [float(i) for i in reg_lambda]
            colsample_bylevel = [float(i) for i in colsample_bylevel]
            colsample_bynode = [float(i) for i in colsample_bynode]
            
            # parameters for the xg boost
            param = {
                "objective" : trial.suggest_categorical('objective', hyperparameters.objective[index]),
                #"scale_pos_weight" : trial.suggest_loguniform('scale_pos_weight', *scale),
                "learning_rate" : trial.suggest_loguniform('learning_rate', *learningrate),
                "gamma" : trial.suggest_int('gamma', *gamma),
                "max_depth" : trial.suggest_int('max_depth', *maxdepth),
                "min_child_weight" : trial.suggest_int('min_child_weight', *minchildweight),
                "subsample" : trial.suggest_loguniform('subsample', *subsample),
                "colsample_bytree" : trial.suggest_loguniform('colsample_bytree', *colsample),
                #"reg_alpha" : trial.suggest_loguniform('reg_alpha', *regalpha),
                #"reg_lambda" : trial.suggest_loguniform('reg_lambda', *reg_lambda),
                "colsample_bylevel" : trial.suggest_loguniform('colsample_bylevel', *colsample_bylevel),
                "colsample_bynode" : trial.suggest_loguniform('colsample_bynode', *colsample_bynode),
            }
            
            # initialize the XG Boost classifier with the passed parameters
            model = XGBClassifier(**param, random_state = int(hyperparameters.sub_sampling_seed[index]))
            
            logging.info("\nIn trial")
            # print the shape of the data
            print_data_info(X_train, Y_train, X_val, Y_val, X_test, Y_test)           
            
            # set a callback for early stopping of bad trials
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
            
            # fit on the data and evaluate on the validation data, auc of roc curve as evaluation metric
            model.fit(X_train,Y_train, eval_set = [(X_val, Y_val)], eval_metric='auc', callbacks = [pruning_callback])             
            
            # set the xg boost in the trial
            trial.set_user_attr(key="best_booster", value=model)
            
            iterations = ['train', 'validation']
            
            # predict and save results for each iteration
            for iteration in iterations:
                
                if iteration == 'train':
                    orig_set = X_train
                    original_labels = Y_train
                    
                elif iteration == 'validation':
                    orig_set = X_val
                    original_labels = Y_val
                
                # predict the probabilites
                raw_preds = model.predict_proba(orig_set)
                
                # retrieve the probability of default
                raw_preds = [i[1] for i in raw_preds]
        
                logging.info("Probability threshold is " + str(pred_threshold))
                print("Probability threshold is", str(pred_threshold))
                
                # set to labels according to predictions
                label_preds = [1 if i>=pred_threshold else 0 for i in raw_preds]
                
                # read the trial number
                f = open("trial_no.txt", "r")
                trial_no = f.read()
                trial_no = int(trial_no)
                f.close()
                
                print('iteration', iteration)
                print('model_num', model_num)
                print('trial_no', trial_no)
                
                # save the metrics
                save_metrics(iteration, model_num, trial_no, original_labels, raw_preds, label_preds)
            
            # read the trial number
            f = open("trial_no.txt", "r")
            trial_no = f.read()
            trial_no = int(trial_no)
            f.close()
            
            # predict on the validation set and store the raw predictions
            raw_preds = model.predict_proba(X_val)
                
            # retrieve the probability of default
            raw_preds = [i[1] for i in raw_preds]
        
            logging.info("Probability threshold is " + str(pred_threshold))
            print("Probability threshold is", str(pred_threshold))
            
            # set to labels according to predictions
            label_preds = [1 if i>=pred_threshold else 0 for i in raw_preds]
            
            # return the score to be maximized
            fpr, tpr, thresholds = roc_curve(Y_val, label_preds)
            auc_from_validation = auc(fpr, tpr)
            
            # compute my new auc
            my_auc = compute_my_auc_roc(Y_val, raw_preds, True, model_num)
            
            return my_auc
    
        print("Defined the function for the trial")
        logging.info("Defined the function for the trial")
        
        # create the study and optimize the xg boost using optuna
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
        
        # optimize and store the best xgb model using a callback
        study.optimize(objective, n_trials=trials,  callbacks=[callback])
        
        # print the best_trial
        print(study.best_trial)
        
        # saving the study history
        filepath = 'study_history/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        hist = study.trials_dataframe()
        name = filepath + "model_" + str(model_num) + '_history.csv'
        hist.to_csv(name, index = False)
        
        # checkpoint to save model weights
        filepath = 'best_trial/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # saving the best trial
        name = filepath + "model_" + str(model_num) + '.txt'
        f = open(name, 'w')
        f.write('\n\nstudy best trial\n\n{}\n'.format(study.best_trial))
        f.close()
        
        # set it to not run since complete
        hyperparameters.loc[index, 'run'] = False
        
        # now retrieving the best model from the best trial of the study, to predict again on the validation and also predict of the test set
        # retrieve the best xgb model
        best_model=study.user_attrs["best_booster"]
        
        results = best_model.evals_result()
        
        filepath = 'final_model_metrics_plots/model_' + str(model_num) + '/'
            # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        x_axis = range(0, len(results['validation_0']['auc']))
        
        # plot auc
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['auc'], label='Validation')
        ax.legend()
        plt.ylabel('AUC ROC')
        plt.title('XGBoost AUC ROC')
        plt.savefig(filepath + 'auc_roc_model_' + str(model_num) + '.png')
        
        # saving the study history
        filepath = 'best_model/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        joblib.dump(best_model, filepath + 'model_' + str(model_num) + '.sav')
        
        # retrieve the auc roc of the validation set from the best trial of the current study
        val_auc_roc_from_best_trial = study.best_value
        
        # predict again on the validation set using the best model
        raw_preds = best_model.predict_proba(X_val)
            
        # retrieve the probability of default
        raw_preds = [i[1] for i in raw_preds]
        
        val_raw = raw_preds
        
        logging.info("Probability threshold is " + str(pred_threshold))
        print("Probability threshold is", str(pred_threshold))
        
        # set to labels according to predictions
        val_label_preds = [1 if i>=pred_threshold else 0 for i in raw_preds]
        
        fpr, tpr, thresholds = roc_curve(Y_val, val_label_preds)
        val_auc_roc_from_best_model = auc(fpr, tpr)
        
        # predict test set using the best model
        raw_preds = best_model.predict_proba(X_test)
        
        # retrieve the probability of default
        raw_preds = [i[1] for i in raw_preds]
        
        logging.info("Probability threshold is " + str(pred_threshold))
        print("Probability threshold is", str(pred_threshold))
        
        # set to labels according to predictions
        test_label_preds = [1 if i>=pred_threshold else 0 for i in raw_preds]
        
        # return the score to be maximized
        fpr, tpr, thresholds = roc_curve(Y_test, test_label_preds)
        test_auc_roc_from_best_model = auc(fpr, tpr)
        
                # return the score to be maximized
        fpr, tpr, thresholds = roc_curve(Y_val, val_raw)
        sklearn_val_auc_roc = auc(fpr, tpr)
        
        # return the score to be maximized
        fpr, tpr, thresholds = roc_curve(Y_test, raw_preds)
        sklearn_test_auc_roc = auc(fpr, tpr)
        
        my_auc_roc_val = compute_my_auc_roc(Y_val, val_raw, False, model_num)
        my_auc_roc_test = compute_my_auc_roc(Y_test, raw_preds, False, model_num)
        
        # save the best model results to a csv            
        save_best_trial_metrics(str(best_model.get_xgb_params()), model_num, val_auc_roc_from_best_model, Y_val, val_label_preds, val_raw, test_auc_roc_from_best_model, Y_test, test_label_preds, raw_preds, my_auc_roc_val, my_auc_roc_test, sklearn_test_auc_roc, sklearn_val_auc_roc)
        
        predictions_df = pd.DataFrame({'loan_status_validation': Y_val, 'pred_prob_validation': val_raw, 'pred_label_validation': val_label_preds})
        predictions_df.to_csv('./predictions/' + str(model_num) + '_validation_predictions.csv', index = False)
        
        predictions_df = pd.DataFrame({'loan_status_test': Y_test, 'pred_prob_test': raw_preds, 'pred_label_test': test_label_preds})
        predictions_df.to_csv('./predictions/' + str(model_num) + '_test_predictions.csv', index = False)
        
        # retrieve the current parameters
        objective_string = hyperparameters.objective[index]
        scale_string = hyperparameters.scale_pos_weight[index]
        learningrate_string = hyperparameters.learning_rate[index]
        gamma_string = hyperparameters.gamma[index]
        maxdepth_string = hyperparameters.max_depth[index]
        minchildweight_string = hyperparameters.min_child_weight[index]
        subsample_string = hyperparameters.subsample[index]
        colsample_string = hyperparameters.colsample_bytree[index]
        regalpha_string = hyperparameters.reg_alpha[index]
        reg_lambda_string = hyperparameters.reg_lambda[index]
        colsample_bylevel = hyperparameters.colsample_bylevel[index]
        colsample_bynode = hyperparameters.colsample_bynode[index]
        
        # convert them to string
        hyperparameters.loc[index, 'objective'] = '[' + ' '.join([str(elem) for elem in objective_string]) + ']'
        hyperparameters.loc[index, 'scale_pos_weight'] = '[' + ' '.join([str(elem) for elem in scale_string]) + ']'
        hyperparameters.loc[index, 'learning_rate'] = '[' + ' '.join([str(elem) for elem in learningrate_string]) + ']'
        hyperparameters.loc[index, 'gamma'] = '[' + ' '.join([str(elem) for elem in gamma_string]) + ']'
        hyperparameters.loc[index, 'max_depth'] = '[' + ' '.join([str(elem) for elem in maxdepth_string]) + ']'
        hyperparameters.loc[index, 'min_child_weight'] = '[' + ' '.join([str(elem) for elem in minchildweight_string]) + ']'
        hyperparameters.loc[index, 'subsample'] = '[' + ' '.join([str(elem) for elem in subsample_string]) + ']'
        hyperparameters.loc[index, 'colsample_bytree'] = '[' + ' '.join([str(elem) for elem in colsample_string]) + ']'
        hyperparameters.loc[index, 'reg_alpha'] = '[' + ' '.join([str(elem) for elem in regalpha_string]) + ']'
        hyperparameters.loc[index, 'reg_lambda'] = '[' + ' '.join([str(elem) for elem in reg_lambda_string]) + ']'
        hyperparameters.loc[index, 'colsample_bylevel'] = '[' + ' '.join([str(elem) for elem in colsample_bylevel]) + ']'
        hyperparameters.loc[index, 'colsample_bynode'] = '[' + ' '.join([str(elem) for elem in colsample_bynode]) + ']'
        
        hyperparameters.to_csv('./hyperparameters/hyperparameters.csv', sep = ',', index = False)

# BM for each training and testing set
for index, row in hyperparameters.iterrows():
    # convert them to string
    hyperparameters.loc[index, 'objective'] = '[' + ' '.join([str(elem) for elem in objective_string]) + ']'
    hyperparameters.loc[index, 'scale_pos_weight'] = '[' + ' '.join([str(elem) for elem in scale_string]) + ']'
    hyperparameters.loc[index, 'learning_rate'] = '[' + ' '.join([str(elem) for elem in learningrate_string]) + ']'
    hyperparameters.loc[index, 'gamma'] = '[' + ' '.join([str(elem) for elem in gamma_string]) + ']'
    hyperparameters.loc[index, 'max_depth'] = '[' + ' '.join([str(elem) for elem in maxdepth_string]) + ']'
    hyperparameters.loc[index, 'min_child_weight'] = '[' + ' '.join([str(elem) for elem in minchildweight_string]) + ']'
    hyperparameters.loc[index, 'subsample'] = '[' + ' '.join([str(elem) for elem in subsample_string]) + ']'
    hyperparameters.loc[index, 'colsample_bytree'] = '[' + ' '.join([str(elem) for elem in colsample_string]) + ']'
    hyperparameters.loc[index, 'reg_alpha'] = '[' + ' '.join([str(elem) for elem in regalpha_string]) + ']'
    hyperparameters.loc[index, 'reg_lambda'] = '[' + ' '.join([str(elem) for elem in reg_lambda_string]) + ']'
    hyperparameters.loc[index, 'colsample_bylevel'] = '[' + ' '.join([str(elem) for elem in colsample_bylevel]) + ']'
    hyperparameters.loc[index, 'colsample_bynode'] = '[' + ' '.join([str(elem) for elem in colsample_bynode]) + ']'

hyperparameters.to_csv('./hyperparameters/hyperparameters.csv', sep = ',', index = False)
os.remove("trial_no.txt")
