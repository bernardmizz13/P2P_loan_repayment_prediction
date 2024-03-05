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
# log file name
__file__ = 'logs/mdp_heuristic_3_log_' + datestring

# initializing the logger
logging.basicConfig(filename=os.path.realpath(__file__) + '.log',
                    filemode='a+',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logging.info("Started the script!")
logging.info("Timestamp is " + datestring)
print("Timestamp is", datestring)

# create directory to store the randomly generated subsets
path = 'datasets/mdp_heuristic_3/' + datestring + '/'

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# for data manipulation
import numpy as np
# for data manipulation
import pandas as pd
# for euclidean distance calculation
from scipy.spatial import distance
# to calculate the distance matrix
from sklearn.metrics.pairwise import euclidean_distances
# for command line arguments
import sys

# start timer
start_time = time.time()

# the desired size of the most distant set
size = sys.argv[1]

# name of the file
file_name = 'test.csv'
# file_name = 'p2p_lendingclub_70_5percent.csv'

# read the training CSV from where we will select the most distant set
p2p_df = pd.read_csv('datasets/p2p_training_set/' + file_name)

# reset the index
p2p_df = p2p_df.reset_index(drop=True)

logging.info('Successfully read the training CSV ' + file_name)
print("Successfully read the training CSV", file_name)

# ###########################################################################################################

# dataframe to store the total distances
distances_df = pd.DataFrame({'record_no': list(range(0, len(p2p_df.index))), 'distance': [0] * len(p2p_df.index)})

logging.info("Calculating distances for each record")
print("Calculating distances for each record")
 
# get the sum of the pairwise distance for each record
distances_df['distance'] = [np.sum(euclidean_distances(record.reshape(1, 134), p2p_df.to_numpy())) for record in p2p_df.to_numpy()]

# ###########################################################################################################

# sort by distance ascending to have the record with the highest distance on top
distances_df_sorted = distances_df.sort_values('distance', ascending=False)

# top [size] records with the highest distance
most_distant_set_indexes = distances_df_sorted.index.tolist()[:int(size)]

# retrieve the most distant records
most_distant_set = p2p_df.loc[most_distant_set_indexes]

# set the indexes
most_distant_set.index = most_distant_set_indexes

# save to csv
most_distant_set.to_csv(path + 'most_distant_set.csv', index = True)

current_time = time.time()
print("--- total time taken so far is %s ---" % ((current_time - start_time)/60))
logging.info("--- total time taken so far is " +  str((current_time - start_time)/60) + ' ---\n')