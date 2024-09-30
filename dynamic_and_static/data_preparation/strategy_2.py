# for time tracking
import time
# for paths and logging
import os
# for logging purposes
import logging
# for file name saving
from datetime import datetime
# get date time string
datestring = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

# create directory to store the randomly generated subsets
path = 'runs/strategy_2/' + datestring + '/'

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# log file name
__file__ = path + 'log' + datestring

# initializing the logger
logging.basicConfig(filename=os.path.realpath(__file__) + '.log',
                    filemode='a+',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logging.info("Started the script!")
logging.info("Timestamp is " + datestring)
print("Timestamp is", datestring)

# for data manipulation
import numpy as np
# for data manipulation
import pandas as pd
# for ceiling function
import math
# for cl arguments
import sys

window_size = int(sys.argv[1])
step = int(sys.argv[2])
train_percentage = float(sys.argv[3])

# TREAT the DATE issue_d columns
def separateDates(dataframe, columns):
    for f in columns:
        dataframe[f] = pd.to_datetime(dataframe[f], format='%b-%Y')
        year = dataframe[f].apply(lambda x: x.strftime('%Y') if not pd.isnull(x) else '')
        month = dataframe[f].apply(lambda x: x.strftime('%m') if not pd.isnull(x) else '')
        dataframe[(f + '_month')] = month
        dataframe[(f + '_year')] = year
        dataframe[(f + '_month')] = pd.to_numeric(dataframe[(f + '_month')])
        dataframe[(f + '_year')] = pd.to_numeric(dataframe[(f + '_year')])
    return dataframe

# #######################################################################################################

print("Window size", str(window_size), ", step", str(step))
logging.info("Window size " + str(window_size) + " , step " + str(step))

# start timer
start_time = time.time()

# read the training and testing set and reset their indexes
train = pd.read_csv('./runs/split_70_30/2021_03_11_19_47_16/p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_70_percent.csv')
train = train.reset_index(drop=True)
test = pd.read_csv('./runs/split_70_30/2021_03_11_19_47_16/p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_30_percent.csv')
test = test.reset_index(drop=True)

num_of_iterations = math.ceil(((len(test.index)-window_size)/step)+1)

print("We will have", str(num_of_iterations), "windows")
logging.info("We will have " + str(num_of_iterations) + " windows")

print("Iterating windows...")
logging.info("Iterating windows...")


# now create separate training and testing set
# a test set consists of a window, and its training set will be the last previous 1% or 5%
no_of_training_records = 20240
if train_percentage == 0.01:
    no_of_training_records = 4050

train_names = []
validation_names = []
test_names = []

# dataframe for issue_d frequency count
df_issue_d_freq = pd.DataFrame({'issue_d': test.issue_d.unique()})
df_issue_d_freq['issue_d_orig'] = df_issue_d_freq['issue_d']

# seperate the dates
df_issue_d_freq = separateDates(df_issue_d_freq, ['issue_d'])

# sort by year and then by month
df_issue_d_freq = df_issue_d_freq.sort_values(['issue_d_year', 'issue_d_month'], ascending=[True, True])

entire_df = train.append(test)

# store all the validation data
all_validation_data = train.tail(window_size)
all_validation_data = all_validation_data.append(test)

# dataframe for train issue_d frequency count
validation_df_issue_d_freq = pd.DataFrame({'issue_d': all_validation_data.issue_d.unique()})
validation_df_issue_d_freq['issue_d_orig'] = validation_df_issue_d_freq['issue_d']

# seperate the dates
validation_df_issue_d_freq = separateDates(validation_df_issue_d_freq, ['issue_d'])

# sort by year and then by month
validation_df_issue_d_freq = validation_df_issue_d_freq.sort_values(['issue_d_year', 'issue_d_month'], ascending=[True, True])

# dataframe for train issue_d frequency count
train_df_issue_d_freq = pd.DataFrame({'issue_d': entire_df.issue_d.unique()})
train_df_issue_d_freq['issue_d_orig'] = train_df_issue_d_freq['issue_d']

# seperate the dates
train_df_issue_d_freq = separateDates(train_df_issue_d_freq, ['issue_d'])

# sort by year and then by month
train_df_issue_d_freq = train_df_issue_d_freq.sort_values(['issue_d_year', 'issue_d_month'], ascending=[True, True])

# #######################################################################################################

df = test

for i in range(0, num_of_iterations):

    to_use = pd.DataFrame()
    
    if len(df.index) >= window_size:
    
        # retrieve the first number of records using the window size
        to_use = df.head(window_size)
        
        if step > 0:
            # remove the first n (step size) rows
            df = df.iloc[step:]
        
    else:
        
        # retrieve the last window as it is since it is smaller than the window size
        to_use = df
    
    # get the frequency count of column issue_d    
    df_issue_d_freq['window_' + str(i)] = df_issue_d_freq['issue_d_orig'].map(to_use['issue_d'].value_counts())
    
    # drop unnecessary columns
    to_use = to_use.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
    to_use.to_csv(path + 'test_' + str(i) + '.csv', index = False)
    
    test_names.append('test_' + str(i))
       
    # create the validation set
    
    validation_set = pd.DataFrame()
    
    # if this is the first set, then the validation is the last set of records from the training set
    if i == 0:
        validation_set = train.tail(window_size)
        
        # get the months in the validation set
        months = validation_set.issue_d.unique()
    
        # get the frequency count of column issue_d    
        validation_df_issue_d_freq['window_' + str(i)] = validation_df_issue_d_freq['issue_d_orig'].map(validation_set['issue_d'].value_counts())
        
        # remove unnecessary features
        validation_set = validation_set.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
    
    # else the validation is the previous test set
    else:
        validation_set = pd.read_csv(path + 'test_' + str(i-1) + '.csv', sep = ',')
        
        validation_df_issue_d_freq['window_' + str(i)] = df_issue_d_freq['window_' + str(i-1)]
          
    # save
    validation_set.to_csv(path + 'validation_' + str(i) + '.csv', index = False)      
    
    validation_names.append('validation_' + str(i))
    
    index = 0
    
    # if first set, then index is the first record of the validation set, since the validation set was taken from the train set, thus the index is valid
    if i == 0:
        index = len(train.index)-window_size
        
    else:
        # get the index of the first test record then deduct the size of the validation sset, since it is from their that the train set will be taken
        index = to_use.index.tolist()[0] - window_size    
    
    new_train = pd.DataFrame()
    
    if i != 0:
        # create the training set
        if index > no_of_training_records:
            new_train = test.iloc[index-no_of_training_records: index]
        
        # else retrieve data from the training set as well
        else:
            # get the number of remaining records to retrieve from the training set
            no_of_records = no_of_training_records - index
            
            # get the first set of records from the training set
            new_train = train.tail(no_of_records)
            
            # append the remaining records from the test set
            # get the first set of training records from the test set
            new_train = new_train.append(test.head(index))
    else:
    
        new_train = train.iloc[len(train.index) - window_size - no_of_training_records : len(train.index) - window_size ]
        
    # get the months in the training set
    months = new_train.issue_d.unique()
    
    # get the frequency count of column issue_d    
    train_df_issue_d_freq['window_' + str(i)] = train_df_issue_d_freq['issue_d_orig'].map(new_train['issue_d'].value_counts())
    
    # drop unnecessary columns
    new_train = new_train.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
    # save
    new_train.to_csv(path + 'train_' + str(i) + '.csv', index = False)
    
    train_names.append('train_' + str(i))
    
    print("Finished window", str(i))
    logging.info("Finished window " + str(i))

# #######################################################################################################

# save the issue_d frequency per window tree to csv
df_issue_d_freq = df_issue_d_freq.drop(['issue_d'], axis = 1)
df_issue_d_freq.to_csv(path + 'test_issue_d_window_freq.csv', index = False)

# save the issue_d frequency per window tree to csv
train_df_issue_d_freq = train_df_issue_d_freq.drop(['issue_d'], axis = 1)
train_df_issue_d_freq.to_csv(path + 'train_issue_d_window_freq.csv', index = False)

# save the issue_d frequency per window tree to csv
validation_df_issue_d_freq = validation_df_issue_d_freq.drop(['issue_d'], axis = 1)
validation_df_issue_d_freq.to_csv(path + 'validation_issue_d_window_freq.csv', index = False)

info = pd.DataFrame({'train_names': train_names, 'validation_names': validation_names, 'test_names': test_names,})
info.to_csv(path + 'info.csv', index = False)

current_time = time.time() 
print("--- total time taken so far in minutes is %s ---" % ((current_time - start_time)/60))
logging.info("--- total time taken so far in minutes is " +  str((current_time - start_time)/60) + ' ---\n')