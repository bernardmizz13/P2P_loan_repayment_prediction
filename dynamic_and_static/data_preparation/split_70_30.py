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
path = 'runs/split_70_30/' + datestring + '/'

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# log file name
__file__ = path + 'split_70_30_log_' + datestring

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
# for euclidean distance calculation
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/constructed_data/p2p_lendingclub_filtered_shuffled_seed_23872.csv')

y = df.loan_status
X = df.drop(['loan_status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
 
X_train['loan_status'] = y_train
X_test['loan_status'] = y_test

X_train.to_csv(path + 'p2p_lendingclub_filtered_shuffled_seed_23872_70_percent.csv', index = False)
X_test.to_csv(path + 'p2p_lendingclub_filtered_shuffled_seed_23872_30_percent.csv', index = False)

logging.info("Successfully split and saved 70% and 30% of the randomly suffled data into training and testing, respectively.")
print("Successfully split and saved 70% and 30% of the randomly suffled data into training and testing, respectively.")

df = pd.read_csv('./data/constructed_data/p2p_lendingclub_filtered_sorted_by_issue_d.csv')

y = df.loan_status
X = df.drop(['loan_status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
 
X_train['loan_status'] = y_train
X_test['loan_status'] = y_test

X_train.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_issue_d_70_percent.csv', index = False)
X_test.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_issue_d_30_percent.csv', index = False)

logging.info("Successfully split and saved 70% and 30% of the data sorted by loan issue date into training and testing, respectively.")
print("Successfully split and saved 70% and 30% of the data sorted by loan issue date into training and testing, respectively.")

df = pd.read_csv('./data/constructed_data/p2p_lendingclub_filtered_sorted_by_id.csv')

y = df.loan_status
X = df.drop(['loan_status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
 
X_train['loan_status'] = y_train
X_test['loan_status'] = y_test

X_train.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_id_70_percent.csv', index = False)
X_test.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_id_30_percent.csv', index = False)

logging.info("Successfully split and saved 70% and 30% of the data sorted by ID into training and testing, respectively.")
print("Successfully split and saved 70% and 30% of the data sorted by ID into training and testing, respectively.")

df = pd.read_csv('./data/constructed_data/p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id.csv')

y = df.loan_status
X = df.drop(['loan_status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
 
X_train['loan_status'] = y_train
X_test['loan_status'] = y_test

X_train.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_70_percent.csv', index = False)
X_test.to_csv(path + 'p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_30_percent.csv', index = False)

X_train_use = X_train.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
X_train_use.to_csv(path + 'use_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_70_percent.csv', index = False)

X_test_use = X_test.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
X_test_use.to_csv(path + 'use_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_30_percent.csv', index = False)

# get the percentage of defaulters
percentage_a = (len(X_train_use[X_train_use.loan_status == 1].index)/len(X_train_use.index)) * 100
print(str(round(percentage_a, 2)))
logging.info(str(round(percentage_a, 2)))

# generated seeds from random.org
seeds = [18434, 26, 9444, 20461, 3852]

a = 20240/len(X_train_use.index)
b = 4050/len(X_train_use.index)

for seed in seeds:
    a_sample_df = pd.DataFrame()
    b_sample_df = pd.DataFrame()

    a_sample_df = X_train_use.sample(frac=a, random_state = seed)
    percentage_a = (len(a_sample_df[a_sample_df.loan_status == 1].index)/len(a_sample_df.index)) * 100
    print(str(round(percentage_a, 2)))
    logging.info(str(round(percentage_a, 2)))
    b_sample_df = X_train_use.sample(frac=b, random_state = seed)
    percentage_a = (len(b_sample_df[b_sample_df.loan_status == 1].index)/len(b_sample_df.index)) * 100
    print(str(round(percentage_a, 2)))
    logging.info(str(round(percentage_a, 2)))
    
    a_sample_df.to_csv(path + 'sorted_by_issue_d_and_then_id_train_data_' + str(seed) + '_20240.csv', index = False)
    b_sample_df.to_csv(path + 'sorted_by_issue_d_and_then_id_train_data_' + str(seed) + '_4050.csv', index = False)

logging.info("Successfully split and saved 70% and 30% of the data sorted by ID into training and testing, respectively.")
print("Successfully split and saved 70% and 30% of the data sorted by ID into training and testing, respectively.")

df = pd.read_csv('./data/constructed_data/p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id.csv')

y = df.loan_status
X = df.drop(['loan_status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
 
X_train['loan_status'] = y_train
X_test['loan_status'] = y_test

X_train.to_csv(path + 'train_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_50_percent.csv', index = False)
X_test.to_csv(path + 'test_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_50_percent.csv', index = False)

X_train_use = X_train.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
X_train_use.to_csv(path + 'use_train_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_50_percent.csv', index = False)

X_test_use = X_test.drop(['issue_d', 'id', 'issue_d_no'], axis = 1)
X_test_use.to_csv(path + 'use_test_p2p_lendingclub_filtered_sorted_by_issue_d_and_then_id_50_percent.csv', index = False)

# get the percentage of defaulters
percentage_a = (len(X_train_use[X_train_use.loan_status == 1].index)/len(X_train_use.index)) * 100
print(str(round(percentage_a, 2)))
logging.info(str(round(percentage_a, 2)))


