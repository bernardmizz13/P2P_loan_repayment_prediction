#### Following https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/ to run the model on the Lending Club data, and also https://machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/, https://www.tensorflow.org/guide/keras/train_and_evaluate and https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/

#### Also following https://machinelearningmastery.com/improve-deep-learning-performance/

#### In this script we create a multilayer perceptron

#### Doing the necessary imports

# for data manipulation
import pandas as pd
import numpy as np
# for plotting curves
import matplotlib.pyplot as plt
# for the model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import keras.metrics as metrics_k
# for optimizers
import keras
# for file name saving
from datetime import datetime
# get date time string
datestring = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
# for saving the model as image
from keras.utils.vis_utils import plot_model
# for evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
# for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for confusion matrix
from sklearn.metrics import confusion_matrix as cf
# for time tracking
import time
# for paths and logging
import os
# for logging purposes
import logging
# for splitting
from sklearn.model_selection import train_test_split
# for output one-hot encoding
from sklearn.preprocessing import LabelEncoder
# for auc
from sklearn.metrics import auc

# folder for confution matrices and predictions
filepath = 'mlp/logs/'
# if model directory does not exist create it
if not os.path.exists(filepath):
    os.makedirs(filepath)

__file__ = 'mlp/logs/mlp_log_' + str(datestring)

# initializing the logger
logging.basicConfig(filename=os.path.realpath(__file__) + '.log',
                    filemode='a+',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging.info("Started the notebook!")

#################################################################################

# define the prediction threshold
pred_threshold = 0.5

def save_results(model_no, f1, precision, recall, accuracy, auc_roc_label, auc_roc_prob, eval_auc_roc, my_auc_roc, tn, fn, tp, fp):
    results = pd.DataFrame({'eval_auc_roc': [eval_auc_roc], 'auc_roc_label': [auc_roc_label], 'auc_roc_prob': [auc_roc_prob], 'my_auc_roc': [my_auc_roc], 'accuracy': [accuracy], 'f1_score': [f1], 'precision': [precision], 'recall': [recall], 'tn': [tn], 'fn': [fn], 'tp': [tp], 'fp': [fp]})
    
    # checkpoint to save model weights
    filepath = 'mlp/final_test_results/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    results.to_csv('mlp/final_test_results/model_' + str(model_no) + '_results.csv', index=False)

#################################################################################

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

#################################################################################

# baseline model
def create_model(index, remove_features, train_epochs, train_batch_size, scaled, activation_function_1, activation_function_2, output_activation_function, model_metric, layer_dropout_1, layer_dropout_2, layer_1, layer_2,
    lr, mlp_optimizer, mlp_loss):
    
    # create model
    model = Sequential()
    
    if activation_function_1 == 'linear':
        if remove_features is True:
            model.add(Dense(layer_1, input_dim=23))
        else:
            model.add(Dense(layer_1, input_dim=133))
    elif 'leaky relu' in activation_function_1:
        if remove_features is True:
            model.add(Dense(layer_1, input_dim=23))
        else:
            model.add(Dense(layer_1, input_dim=133))
        
        alp = activation_function_1.replace("leaky relu ", "")
        model.add(LeakyReLU(alpha = float(alp)))
    else:
        if remove_features is True:
            model.add(Dense(layer_1, input_dim=23, activation=activation_function_1))
        else:
            model.add(Dense(layer_1, input_dim=133, activation=activation_function_1))
        
    if layer_dropout_1 != 0:
        model.add(Dropout(layer_dropout_1))
    
    if activation_function_2 == 'linear':
        model.add(Dense(layer_2))
    elif 'leaky relu' in activation_function_2:
        model.add(Dense(layer_2))
        
        alp = activation_function_2.replace("leaky relu ", "")
        model.add(LeakyReLU(alpha = float(alp)))
    else:
        model.add(Dense(layer_2, activation=activation_function_2))
    
    if layer_dropout_2 != 0:
        model.add(Dropout(layer_dropout_2))
    
    # final layer
    model.add(Dense(1, activation=output_activation_function)) 

    # initialize the adam optimizer with the learning rate
    optim = None
    if mlp_optimizer == 'Adam':
        optim = keras.optimizers.Adam(learning_rate=lr)
        
    # Compile model
    if model_metric == 'AUC':
        model.compile(loss=mlp_loss, optimizer=optim, metrics=[metrics_k.AUC(name = 'auc_roc')])
    elif model_metric == 'accuracy':
        model.compile(loss=mlp_loss, optimizer=optim, metrics=[model_metric])
    else:
        model.compile(loss=mlp_loss, optimizer=optim, metrics=[metrics_k.AUC(name = 'auc_roc'), metrics_k.Accuracy(name = 'accuracy')])
    model.summary()
    
    # plot_model(model, to_file='mlp/mlp_structures/' + str(index) + '_model.png')
    
    return model

#################################################################################

# function that computes my auc roc
def compute_my_auc_roc(orig_labels, predict_probs, name, in_epoch):
    
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
        
    plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    
    filepath = ''
    
    if in_epoch is False:
        # folder for confution matrices and predictions
        filepath = 'mlp/final_model_my_auc_roc_curves/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    else:
        # folder for confution matrices and predictions
        filepath = 'mlp/epoch_my_auc_roc_curves/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    
    plt.savefig(filepath + name + '.png')
    plt.close()
    
    return auc

#################################################################################

def save_model_results(model_no, epoch, results, train_loss, val_loss, test_loss, train_eval_auc, val_eval_auc, test_eval_auc):
    
    # dataframe for results
    results_df = pd.DataFrame(columns = ['model_no', 'epoch', 'iteration', 'loss', 'eval_auc_roc', 'auc_roc_label', 'auc_roc_prob', 'my_auc_roc', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn'])
    
    # checkpoint to save model weights
    filepath = 'mlp/epoch_metric_results/'
    # if model directory does not exist create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    print('Epoch is', str(epoch + 1))
    
    # if first epoch create csv, (on cmd it is printed as 1 however internally it is 0)
    if epoch == 0:
        results_df.to_csv(filepath + 'model_' + str(model_no) + '_metrics.csv', index = False)
    else:
        results_df = pd.read_csv(filepath + 'model_' + str(model_no) + '_metrics.csv')
    
    for i in range(0, 3):
        
        j=0
        loss = 0
        eval_auc_roc = 0
        iteration = ''
        if i == 0:
            j = 0
            iteration = 'train'
            loss = train_loss
            eval_auc_roc = train_eval_auc
        elif i == 1:
            j = 2
            iteration = 'validation'
            loss = val_loss
            eval_auc_roc = val_eval_auc
        else:
            j = 4
            iteration = 'test'
            loss = test_loss
            eval_auc_roc = test_eval_auc
        
        # convert list of lists to one list
        results[j+1] = [item for sublist in results[j+1] for item in sublist]
        
        # get the predicted probabilies
        prob_predictions = results[j+1]
        
        # since model might not fully learn at early epochs, set a treshold for values greater than 0.5 to be set as defaulted loans, non default otherwise
        # will work if predictions are 1s and 0s
        predictions = [1 if i>=pred_threshold else 0 for i in prob_predictions]
        # logging.info("Predictions for " + iterations + " set in after epoch " + str(epoch) + " for model " + str(model_no) + " are continuous and not binary!")
           
        # checkpoint to save model weights
        filepath = 'mlp/epoch_predictions/model_'+str(model_no)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # store original labels
        orig_labels = results[j]
        
        print('List lengths', len(orig_labels))
        print('List lengths', len(prob_predictions))
        print('List lengths', len(predictions))
        
        df = pd.DataFrame({'y_Actual': orig_labels, 'y_Predicted_probabilities': prob_predictions, 'y_Predicted_labels': predictions})
        df.to_csv(filepath + "/model_" + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration + '_predictions.csv', index = False)
           
        # checkpoint to save model weights
        filepath = 'mlp/epoch_confusion_matrices/model_'+str(model_no)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        logging.info(iteration + " size: " + str(len(predictions)) + ", " + str(len(results[j])))
        
        # produce the confusion matrix
        conf_mat = cf(orig_labels, list(predictions))
        name = filepath + "/model_" + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration + '_sklearn_confusion_matrix.txt'
        f = open(name, 'w')
        f.write('\n\nConfusion Matrix\n\n{}\n'.format(conf_mat))
        f.close()   
        
        pandas_conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted_labels'], rownames=['Actual'], colnames=['Predicted'])
        name = filepath + "/model_" + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration + '_pandas_confusion_matrix.txt'
        f = open(name, 'w')
        f.write('\n\nConfusion Matrix\n\n{}\n'.format(pandas_conf_matrix))
        f.close()
        
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        tp = conf_mat[1][1]
        
        # get the fpr, tpr and threshold
        fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(orig_labels, prob_predictions)
        auc_dnn_prob = auc(fpr_dnn, tpr_dnn)
        
        fpr, tpr, thresholds = roc_curve(orig_labels, predictions)
        auc_labels = auc(fpr, tpr)
        
        sk_acc = accuracy_score(orig_labels, predictions)
        sk_prec = precision_score(orig_labels, predictions)
        sk_rec = recall_score(orig_labels, predictions)
        sk_f1 = f1_score(orig_labels, predictions)        
        
        # print
        print('\n\n')
        print(iteration)
        print('epoch ', str(epoch + 1))
        print('sklearn precision ', sk_prec)
        print('sklearn recall ', sk_rec)
        print('sklearn accuracy ', sk_acc)
        print('sklearn f1 score ', sk_f1)
        print('\n\n')
        
        logging.info('\n\n')
        logging.info(iteration)
        logging.info('epoch ' + str(epoch + 1))
        logging.info('sklearn precision ' + str(sk_prec))
        logging.info('sklearn recall ' + str(sk_rec))
        logging.info('sklearn accuracy ' + str(sk_acc))
        logging.info('sklearn f1 score ' + str(sk_f1))
        logging.info('\n\n')
        
        # checkpoint to save model weights
        filepath = 'mlp/epoch_auc_roc_curve_plots/model_'+str(model_no)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # plot the auc roc curve graph for the labels
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc_labels))
        plt.title('ROC curve')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.savefig(filepath + '/model_' + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration + '_labels_AUC_ROC.png')
        plt.close()
        
        # plot the auc roc curve graph for the probabilities
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_dnn, tpr_dnn, label='area = {:.3f}'.format(auc_dnn_prob))
        plt.title('ROC curve')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.savefig(filepath + '/model_' + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration + '_prob_AUC_ROC.png')
        plt.close()
        
        precision = sk_prec
        accuracy = sk_acc
        recall = sk_rec
        f1 = sk_f1
        
        # compute our AUC ROC
        my_auc_roc = compute_my_auc_roc(orig_labels, prob_predictions, 'model_' + str(model_no) + '_epoch_' + str(epoch) + '_' + iteration, True)
        
        # add the new metrics
        results_df = results_df.append(pd.DataFrame({'model_no': [model_no], 'epoch': [epoch], 'iteration': [iteration], 'loss': [loss], 'eval_auc_roc': [eval_auc_roc], 'auc_roc_label': [auc_labels], 'auc_roc_prob': [auc_dnn_prob], 'my_auc_roc': [my_auc_roc],
                                                    'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1_score': [f1], 'tp': [tp], 'fp': [fp], 'tn': [tn], 'fn': [fn]}))
        # save to csv
        results_df.to_csv('mlp/epoch_metric_results/' + 'model_' + str(model_no) + '_metrics.csv', index = False)
        
#################################################################################

# create a callback to evaluate and predict with the model at each epoch
class evaluate_after_epoch(Callback):
    
    def __init__(self, model_no, x_test, y_test, x_val, y_val, x_train, y_train):
        self.x_test = x_test
        self.y_test = y_test
        # BM TEST
        self.x_val = x_val
        self.y_val = y_val
        self.x_train = x_train
        self.y_train = y_train
        self.model_no = model_no
    
    def on_epoch_end(self, epoch, logs=None):
        # store the datasets
        x_train = self.x_train
        y_train = self.y_train
        x_val = self.x_val
        y_val = self.y_val
        x_test = self.x_test
        y_test = self.y_test
        
        # convert to list
        # y_val = y_val.tolist()
        # y_val = [item for sublist in y_val for item in sublist]
        
        # evaluate to capture loss
        train_evaluation_results = self.model.evaluate(x_train, y_train)
        val_evaluation_results = self.model.evaluate(x_val, y_val)
        test_evaluation_results = self.model.evaluate(x_test, y_test)
        
        # make predictions on train, validation and test data
        print("\nEvaluating train")
        print(x_train.head(5))
        print('Epoch train shape: ' + str(x_train.shape))
        logging.info('Epoch train shape: ' + str(x_train.shape))
        train_predictions = self.model.predict(x_train)
        
        print("\nEvaluating validation")
        print(x_val.head(5))
        logging.info('Epoch validation shape: ' + str(x_val.shape))
        print('Epoch validation shape: ' + str(x_val.shape))
        val_predictions = self.model.predict(x_val)
        
        print("\nEvaluating test")
        print(x_test.head(5))
        logging.info('Epoch test shape: ' + str(x_test.shape))
        print('Epoch test shape: ' + str(x_test.shape))
        test_predictions = self.model.predict(x_test)
        
        save_model_results(self.model_no, epoch, [y_train.values.tolist(), train_predictions, y_val.values.tolist(), val_predictions, y_test.values.tolist(), test_predictions], train_evaluation_results[0], val_evaluation_results[0], test_evaluation_results[0], train_evaluation_results[1], val_evaluation_results[1], test_evaluation_results[1])
                          
#################################################################################

# read the hyper parameters file
hyperparameters_and_results = pd.read_csv('mlp/hyperparameters_and_final_results/hyperparameters.csv')
logging.info("Successfully read the hyperparameters CSV file!")

#################################################################################

# start timer
start_time = time.time()
logging.info("Will start running the MLPs!")
# iterate over hyperparameters
for index, row in hyperparameters_and_results.iterrows():
    
    # if selected to run
    if row['run'] is True:
    
        # set the parameters
        scaled = row['scaled']
        train_epochs = row['epochs']
        train_batch_size = row['batch_size']
        activation_function_1 = row['activation_function_1']
        activation_function_2 = row['activation_function_2']
        output_activation_function = row['output_activation_function']
        model_metric = row['model_metric']
        drop_1 = row['dropout_1']
        drop_2 = row['dropout_2']
        layer_1 = row['layer_1']
        layer_2 = row['layer_2']
        learn_rate = row['learning_rate']
        val_random = row['val_random']
        mlp_optimizer = row['optimizer']
        mlp_loss = row['loss']
        
        train_file = row['train_data'] + '.csv'
        validation_file = row['validation_data'] + '.csv'
        test_file = row['test_data'] + '.csv'
        
        # load train dataset
        dataframe = pd.read_csv('data/' + train_file)
        dataframe.loan_status = dataframe.loan_status.astype(int)
        # dataframe.to_csv(row['train_data'] + '_saved.csv', sep = ',', index = False)
        # dataframe = pd.read_csv(row['train_data'] + '_saved.csv', sep = ',')
        
        # split into input (X) and output (Y) variables
        Y_train = dataframe.loan_status
        X_train = dataframe.drop(['loan_status'], axis = 1)

        # get the test data
        dataframe = pd.read_csv('data/' + test_file)
        dataframe.loan_status = dataframe.loan_status.astype(int)
        # dataframe.to_csv(row['test_data'] + '_saved.csv', sep = ',', index = False)
        # dataframe = pd.read_csv(row['test_data'] + '_saved.csv', sep = ',')
        
        # split into input (X) and output (Y) variables
        Y_test = dataframe.loan_status
        X_test = dataframe.drop(['loan_status'], axis = 1)

        print("Model no.", str(index))
        logging.info("Model no. " + str(index))

        print('Train and test data read!')
        logging.info('Train and test data read!')
        
        X_val = pd.DataFrame()
        Y_val = pd.DataFrame()
        
        if val_random is True:
            # select random validation set
            # split train in validation and train      
            full_train = X_train
            full_train['loan_status'] = Y_train
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
            val_set = pd.read_csv('data/' + validation_file)
            val_set.loan_status = val_set.loan_status.astype(int)
            # val_set.to_csv(row['validation_data'] + '_saved.csv', sep = ',', index = False)
            # val_set = pd.read_csv(row['validation_data'] + '_saved.csv', sep = ',')     
            
            # store the validation sets
            Y_val = val_set.loan_status
            X_val = val_set.drop(['loan_status'], axis = 1)
            
            logging.info('Validation set not chosen randomly, read from file!')
        
        # drop features if asked to
        if row['remove_features'] is True:
            X_train = X_train.drop(features_to_remove, axis = 1)
            X_val = X_val.drop(features_to_remove, axis = 1)
            X_test = X_test.drop(features_to_remove, axis = 1)
            logging.info('Categorical features removed!')
        else:
            logging.info('Categorical features not removed!')
        
        scaled_or_not = 'non_scaled'
        
        if scaled is True:
            scaled_or_not = 'scaled'

            # Standardize features by removing the mean and scaling to unit variance, mean of 0 and standard deviation of 1
            
            '''Scale values of numerical columns'''
            df_columns = X_train.columns.tolist()
            columns_to_scale = list(set(df_columns) - set(features_to_remove))
            
            scaler = StandardScaler(copy=False)
            
            X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
            X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
            X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
            
            # X_train.to_csv('p2p_lendingclub_70_train_scaled.csv', sep = ',', index = False)
            # X_test.to_csv('p2p_lendingclub_30_test_scaled.csv', sep = ',', index = False)
            # 
            # # re-read for windows format purposes
            # X_train = pd.read_csv("p2p_lendingclub_70_train_scaled.csv", sep = ',')
            # X_test = pd.read_csv("p2p_lendingclub_30_test_scaled.csv", sep = ',')
            
            logging.info('Data scaled and saved for review!')
        else:
            logging.info('Data not scaled!')
        
        logging.info('Train epochs: ' + str(train_epochs))
        logging.info('Train batch size: ' + str(train_batch_size))
        logging.info('activation_function_1: ' + activation_function_1)
        logging.info('activation_function_2: ' + activation_function_2)
        logging.info('output_activation_function: ' + output_activation_function)
        logging.info('dropout_1: ' + str(drop_1))
        logging.info('dropout_2: ' + str(drop_2))
        logging.info('learning_rate: ' + str(learn_rate))
        logging.info('layer_1 size: ' + str(layer_1))
        logging.info('layer_2 size: ' + str(layer_2))
        logging.info('optimizer: ' + (mlp_optimizer))

        logging.info('Train shape: ' + str(X_train.shape))
        logging.info(str(Y_train.shape))
        logging.info('Validation shape: ' + str(X_val.shape))
        logging.info(str(Y_val.shape))
        logging.info('Test shape: ' + str(X_test.shape))
        logging.info(str(Y_test.shape))
        
        print('Train epochs: ' + str(train_epochs))
        print('Train batch size: ' + str(train_batch_size))
        print('activation_function_1: ' + activation_function_1)
        print('activation_function_2: ' + activation_function_2)
        print('output_activation_function: ' + output_activation_function)
        print('dropout_1: ' + str(drop_1))
        print('dropout_2: ' + str(drop_2))
        print('learning_rate: ' + str(learn_rate))
        print('layer_1 size: ' + str(layer_1))
        print('layer_2 size: ' + str(layer_2))
        print('optimizer: ' + (mlp_optimizer))

        print('Train shape: ' + str(X_train.shape))
        print(str(Y_train.shape))
        print('Validation shape: ' + str(X_val.shape))
        print(str(Y_val.shape))
        print('Test shape: ' + str(X_test.shape))
        print(str(Y_test.shape))        

        logging.info('Training the model...')
        print('Training the model...')
        
        # create and train
        mlp = create_model(index, row['remove_features'], train_epochs, train_batch_size, row['scaled'], activation_function_1, activation_function_2, output_activation_function, model_metric, drop_1, drop_2,
            layer_1, layer_2, learn_rate, mlp_optimizer, mlp_loss)
        
        # checkpoint to save model weights
        checkpoint_filepath = 'mlp/epoch_mlp_models/model_'+str(index)+'/'
        # if model directory does not exist create it
        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)
        
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath+'model.epoch_{epoch:02d}_val_loss_{val_loss:.2f}.hdf5',
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=False)
            
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        
        history  = mlp.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=train_epochs, batch_size=train_batch_size,
                           callbacks=[model_checkpoint_callback, evaluate_after_epoch(model_no = index, x_test=X_test, y_test=Y_test, x_val = X_val, y_val = Y_val, x_train=X_train, y_train=Y_train)])
        
        logging.info('MLP trained!')

        # checkpoint to save model weights
        filepath = 'mlp/final_history_plots/model_'+str(index)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        if model_metric == 'accuracy':
            # plot training history
            plt.figure(figsize=(14, 6), dpi=80)
            plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_accuracy'], label='validation')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.savefig(filepath + 'model_' + str(index) + '_history_accuracy.png')
            plt.close()

        elif model_metric == 'AUC':
            plt.figure(figsize=(14, 6), dpi=80)
            plt.plot(history.history['auc_roc'], label='train')
            plt.plot(history.history['val_auc_roc'], label='validation')
            plt.xlabel('epochs')
            plt.ylabel('auc roc')
            plt.legend()
            plt.savefig(filepath + 'model_' + str(index) + '_history_auc.png')
            plt.close()
        
        else:
            # plot training history
            plt.figure(figsize=(14, 6), dpi=80)
            plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_accuracy'], label='validation')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.savefig(filepath + 'model_' + str(index) + '_history_accuracy.png')
            plt.close()
            
            plt.figure(figsize=(14, 6), dpi=80)
            plt.plot(history.history['auc_roc'], label='train')
            plt.plot(history.history['val_auc_roc'], label='validation')
            plt.xlabel('epochs')
            plt.ylabel('auc roc')
            plt.legend()
            plt.savefig(filepath + 'model_' +  str(index) + '_history_auc.png')
            plt.close()
        
        plt.figure(figsize=(14, 6), dpi=80)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(filepath + 'model_' + str(index) + '_history_loss.png')
        plt.close()
        
        logging.info('Evaluating the model on train, validation and test data...')
        
        # evaluate the model
        train_evaluation_results = mlp.evaluate(X_train, Y_train)
        val_evaluation_results = mlp.evaluate(X_val, Y_val)
        test_evaluation_results = mlp.evaluate(X_test, Y_test)
        
        logging.info('MLP evaluated!')

        if model_metric == 'accuracy':
            logging.info("train loss, train acc: " + str(train_evaluation_results[0]) + ' ' + str(train_evaluation_results[1]))
            logging.info("validation loss, validation acc: " + str(val_evaluation_results[0]) + ' ' + str(val_evaluation_results[1]))
            logging.info("test loss, test acc: " + str(test_evaluation_results[0]) + ' ' + str(test_evaluation_results[1]))
        elif model_metric == 'AUC':
            logging.info("train loss, train auc_roc: " + str(train_evaluation_results[0]) + ' ' + str(train_evaluation_results[1]))
            logging.info("validation loss, validation auc_roc: " + str(val_evaluation_results[0]) + ' ' + str(val_evaluation_results[1]))
            logging.info("test loss, test auc_roc: " + str(test_evaluation_results[0]) + ' ' + str(test_evaluation_results[1]))
        else:
            logging.info("train loss, train auc_roc, train accuracy: " + str(train_evaluation_results[0]) + ' ' + str(train_evaluation_results[1]) + ' ' + str(train_evaluation_results[2]))
            logging.info("validation loss, validation auc_roc, validation accuracy: " + str(val_evaluation_results[0]) + ' ' + str(val_evaluation_results[1]) + ' ' + str(val_evaluation_results[2]))
            logging.info("test loss, test auc_roc, test accuracy: " + str(test_evaluation_results[0]) + ' ' + str(test_evaluation_results[1]) + ' ' + str(test_evaluation_results[2]))

        logging.info('Making predictions on test data...')
        
        # make predictions on test set
        y_pred = mlp.predict(X_test)
        
        logging.info('Predictions on test set done!')

        # convert to a list
        y_true = Y_test.values.tolist()
        
        # convert to list
        y_pred = [item for sublist in y_pred for item in sublist]
        y_prob = y_pred
        y_pred = [1 if i>=pred_threshold else 0 for i in y_prob]
        
        # checkpoint to save model weights
        filepath = 'mlp/final_test_predictions/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        df = pd.DataFrame({'y_Actual': y_true, 'y_Predicted_probability': list(y_prob), 'y_Predicted_label': y_pred})        
        df.to_csv(filepath + "/model_" + str(index) + '_final_predictions.csv', index = False)
        
        # checkpoint to save model weights
        filepath = 'mlp/final_test_confusion_matrices/model_'+str(index)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        pandas_conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted_label'], rownames=['Actual'], colnames=['Predicted'])
        name = filepath + "/model_" +  str(index) + '_pandas_final_confusion_matrix.txt'
        f = open(name, 'w')
        f.write('\n\nConfusion Matrix\n\n{}\n'.format(pandas_conf_matrix))
        f.close()
        
        # produce the confusion matrix
        conf_mat = cf(y_true, y_pred)
        name = filepath + "/model_" + str(index) + '_sklearn_final_confusion_matrix.txt'
        f = open(name, 'w')
        f.write('\n\nConfusion Matrix\n\n{}\n'.format(conf_mat))
        f.close()
        
        tn = conf_mat[0][0]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        tp = conf_mat[1][1]
           
        # checkpoint to save model weights
        filepath = 'mlp/final_test_auc_roc_curve_plots/model_'+str(index)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # retrieve the false positive and true positives
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_labels = auc(fpr, tpr)
        # plot the auc roc curve graph
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc_labels))
        plt.title('AUC curve')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.savefig(filepath + '/model_' + str(index) + '_final_AUC_ROC_label.png')
        plt.close()
        
        # retrieve the false positive and true positives
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_prob = auc(fpr, tpr)
        # plot the auc roc curve graph
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc_prob))
        plt.title('ROC curve')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.savefig(filepath + '/model_' + str(index) + '_final_AUC_ROC_prob.png')
        plt.close()
        
        sk_acc = accuracy_score(y_true, y_pred)
        sk_prec = precision_score(y_true, y_pred)
        sk_rec = recall_score(y_true, y_pred)
        sk_f1 = f1_score(y_true, y_pred)
        
        accuracy = sk_acc
        precision = sk_prec
        recall = sk_rec
        f1 = sk_f1
        
        # compute my auc roc
        my_auc_roc = compute_my_auc_roc(y_true, y_prob, 'model_' + str(index) + '_my_auc_roc_plot_test', False)
        
        # save the results to a sperate csv
        save_results(index, f1, precision, recall, accuracy, auc_labels, auc_prob, test_evaluation_results[1], my_auc_roc, tn, fn, tp, fp)

        # save the results into the hyperparameter csv
        hyperparameters_and_results.loc[index, 'accuracy'] = accuracy
        hyperparameters_and_results.loc[index, 'precision'] = precision
        hyperparameters_and_results.loc[index, 'recall'] = recall
        hyperparameters_and_results.loc[index, 'f1_score'] = f1
        hyperparameters_and_results.loc[index, 'eval_auc_roc'] = test_evaluation_results[1]
        hyperparameters_and_results.loc[index, 'auc_roc_label'] = auc_labels
        hyperparameters_and_results.loc[index, 'auc_roc_prob'] = auc_prob
        hyperparameters_and_results.loc[index, 'my_auc_roc'] = my_auc_roc
        
        hyperparameters_and_results.loc[index, 'tp'] = tp
        hyperparameters_and_results.loc[index, 'fp'] = fp
        hyperparameters_and_results.loc[index, 'tn'] = tn
        hyperparameters_and_results.loc[index, 'fn'] = fn
        
        # make predictions on test set
        val_pred = mlp.predict(X_val)
        
        logging.info('Predictions on test set done!')

        # convert to a list
        val_true = Y_val.values.tolist()
        
        # convert to list
        val_pred = [item for sublist in val_pred for item in sublist]
        val_prob = val_pred
        val_pred = [1 if i>=pred_threshold else 0 for i in val_prob]
        
        val_tn = conf_mat[0][0]
        val_fp = conf_mat[0][1]
        val_fn = conf_mat[1][0]
        val_tp = conf_mat[1][1]
        
        sk_acc = accuracy_score(val_true, val_pred)
        sk_prec = precision_score(val_true, val_pred)
        sk_rec = recall_score(val_true, val_pred)
        sk_f1 = f1_score(val_true, val_pred)
        
        val_accuracy = sk_acc
        val_precision = sk_prec
        val_recall = sk_rec
        val_f1 = sk_f1
        
        # retrieve the false positive and true positives
        fpr, tpr, _ = roc_curve(val_true, val_pred)
        val_auc_labels = auc(fpr, tpr)
        
        fpr, tpr, _ = roc_curve(val_true, val_prob)
        val_auc_prob = auc(fpr, tpr)
        
        # compute my auc roc
        val_my_auc_roc = compute_my_auc_roc(val_true, val_prob, 'model_' + str(index) + '_my_auc_roc_plot_validation', False)
        
        # save the results into the hyperparameter csv
        hyperparameters_and_results.loc[index, 'val_accuracy'] = val_accuracy
        hyperparameters_and_results.loc[index, 'val_precision'] = val_precision
        hyperparameters_and_results.loc[index, 'val_recall'] = val_recall
        hyperparameters_and_results.loc[index, 'val_f1_score'] = val_f1
        hyperparameters_and_results.loc[index, 'val_eval_auc_roc'] = val_evaluation_results[1]
        hyperparameters_and_results.loc[index, 'val_auc_roc_label'] = val_auc_labels
        hyperparameters_and_results.loc[index, 'val_auc_roc_prob'] = val_auc_prob
        hyperparameters_and_results.loc[index, 'val_my_auc_roc'] = val_my_auc_roc
        
        hyperparameters_and_results.loc[index, 'val_tp'] = val_tp
        hyperparameters_and_results.loc[index, 'val_fp'] = val_fp
        hyperparameters_and_results.loc[index, 'val_tn'] = val_tn
        hyperparameters_and_results.loc[index, 'val_fn'] = val_fn      
        
        # mark that finished
        hyperparameters_and_results.loc[index, 'run'] = False

        logging.info('accuracy: ' + str(accuracy) + ', precision: ' + str(precision) + ', recall: ' + str(recall) + ', f1-score: ' + str(f1) + ', auc roc label: ' + str(auc_labels) + ', auc roc prob: ' + str(auc_prob) + ', auc roc eval: ' + str(test_evaluation_results[1]))
        
        # after fully training, record the best model epoch
        # read the epoch metrics
        # decide according to loss or auc roc
        epoch_metric = 'loss'
        model_epoch_metrics = pd.read_csv('mlp/epoch_metric_results/model_' + str(index) + '_metrics.csv')
        
        # plot the train, validation and test accuracy, loss and auc roc at each epoch
        iterations = ['train', 'validation', 'test']
        metrics = ['loss', 'eval_auc_roc', 'auc_roc_label', 'auc_roc_prob', 'my_auc_roc', 'accuracy']
        
        # checkpoint to save model weights
        filepath = 'mlp/epoch_metric_plots/model_'+str(index)+'/'
        # if model directory does not exist create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # plot
        for metric in metrics:
            plt.figure(figsize=(14, 6), dpi=80)
            for iteration in iterations:
                model_metrics = model_epoch_metrics.loc[model_epoch_metrics['iteration'] == iteration]
                # do the plotting
                plt.plot(model_metrics[metric].tolist(), label=iteration)
            # close the figure
            plt.xlabel('epochs')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(filepath + 'model_' + str(index) + '_epoch_' + metric + '.png')
            plt.close()
        
        model_best_metrics = pd.DataFrame()
        
        # get the validation metrics only
        model_epoch_metrics = model_epoch_metrics.loc[model_epoch_metrics['iteration'] == 'validation']
        
        # saving the best model
        if epoch_metric == 'loss':
            # get the records with lowest loss
            model_best_metrics = model_epoch_metrics[model_epoch_metrics.loss == model_epoch_metrics.loss.min()]
            logging.info("Selected epoch with lowest loss!")
             # if more than one record is returned
            if len(model_best_metrics.index) > 1:
                # get the records with highest auc roc
                model_best_metrics = model_best_metrics[model_best_metrics.auc_roc_label == model_best_metrics.auc_roc_label.max()]
                logging.info("Selected epoch with highest AUC ROC!")
        elif epoch_metric == 'auc_roc':
            # get the records with highest auc roc
            model_best_metrics = model_epoch_metrics[model_epoch_metrics.auc_roc_label == model_epoch_metrics.auc_roc_label.max()]
            logging.info("Selected epoch with highest AUC ROC!")
             # if more than one record is returned
            if len(model_best_metrics.index) > 1:
                # get the records with lowest loss
                model_best_metrics = model_best_metrics[model_best_metrics.loss == model_best_metrics.loss.min()]
                logging.info("Selected epoch with lowest loss!")
        
        # if more than one record is returned
        if len(model_best_metrics.index) > 1:
            # get the records with highest accuracy
            model_best_metrics = model_best_metrics[model_best_metrics.accuracy == model_best_metrics.accuracy.max()]
            logging.info("Selected epoch with highest accuracy!")
        # if more than one record is returned
        if len(model_best_metrics.index) > 1:
            logging.info("Size is " + str(len(model_best_metrics.index)))
            logging.info("Selected the smallest epoch!")
            # get the records with lowest epoch
            model_best_metrics = model_best_metrics[model_best_metrics.epoch == model_best_metrics.epoch.min()]
            
        # if more than one record is returned
        if len(model_best_metrics.index) > 1:
            model_best_metrics = model_best_metrics.sample(random_state = 1679) 
            logging.info('Finally randomly selected the best model!')
            print('Finally randomly selected the best model!')
        
        # save the best metrics
        print(model_best_metrics)
        hyperparameters_and_results.loc[index, 'best_epoch'] = model_best_metrics['epoch'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_accuracy'] = model_best_metrics['accuracy'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_loss'] = model_best_metrics['loss'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_auc_roc_label'] = model_best_metrics['auc_roc_label'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_auc_roc_prob'] = model_best_metrics['auc_roc_prob'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_eval_auc_roc'] = model_best_metrics['eval_auc_roc'].tolist()[0]
        hyperparameters_and_results.loc[index, 'best_my_auc_roc'] = model_best_metrics['my_auc_roc'].tolist()[0]
                
        # save to csv
        hyperparameters_and_results.to_csv('mlp/hyperparameters_and_final_results/hyperparameters.csv', index = False)

        print('Results saved!\n')
        logging.info('Results saved!')

        print('Model no.', index, 'trained and tested!')
        logging.info('Model no. ' + str(index) +  ' trained and tested!')

        current_time = time.time()

        print("--- total time taken so far in minutes is %s ---" % ((current_time - start_time)/60))
        logging.info("--- total time taken so far in minutes is " +  str((current_time - start_time)/60) + ' ---\n')
        
#################################################################################

# save to csv
hyperparameters_and_results.to_csv('mlp/hyperparameters_and_final_results/hyperparameters.csv', index = False)

logging.info("Saved the results in the hyperparameters CSV file!")

current_time = time.time()

print("--- total script execution time in hours is %s ---" % ((current_time - start_time)/(60*60)))
logging.info("--- total script execution time in hours is " +  str((current_time - start_time)/(60*60)) + ' ---')
logging.info("\n")