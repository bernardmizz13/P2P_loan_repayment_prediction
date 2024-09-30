from imblearn.over_sampling import SMOTE
from imbalance import CsvUtils
from sklearn import tree, linear_model
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss, ClusterCentroids
from imblearn.over_sampling import ADASYN
from imbalance.classifyCrossValidation import ClassifyCV
import pandas as pd
import time, datetime

import _pickle as cPickle

#BM
NUM_PARALLEL_EXEC_UNITS = 6

if __name__ == '__main__':
    # BM read the training and testing set names
    info = pd.read_csv('./data/train_sets.csv')
    
    all_results = pd.DataFrame()
    
    # BM for each training and testing set
    for index, row in info.iterrows():
        
        print("Starting for loop")
        
        # BM
        trainSets = ['./data/' + row['train_names'] + '.csv']
        
        testSets = pd.read_csv('./data/test_sets.csv')
        testSets = testSets.test_names.tolist()
        
        classe = 'loan_status'
        # defino a lista de classificadores
        clfs = [linear_model.LogisticRegression()]
        # clfs = [GaussianNB(), linear_model.LogisticRegression()]
        names = ["Logistic Regression"]
        #names = ["Naive Bayes", "Logistic Regression"]
        final_names = list()
        for set in testSets:
            for name in names:
                final_names.append(str(name+'_'+set[:-4]))

        # defino a lista de tecnicas de sampling
        sTechniques = [SMOTE(random_state=1)]
        technique_names = ["SM_BL_2"]


        def getParamsReSampling(reSamplingTechnique):
        
            if type(reSamplingTechnique) is SMOTE:
                return dict(smt__kind=["borderline2"],
                            smt__ratio=[0.8, 0.9, 1.0], smt__k_neighbors=[1, 3, 5, 7])
            else:
                return dict()

        def getParamsClassifier(classifierName):
            if classifierName == "Decision Tree":
                return dict(clf__criterion=["gini", "entropy"],
                            clf__splitter=["best", "random"],
                            clf__min_samples_split=[2, 10, 20],
                            clf__max_depth=[None, 2, 5, 10],
                            clf__min_samples_leaf=[1, 5, 10],
                            clf__max_leaf_nodes=[None, 5, 10, 20])
            elif classifierName == "Naive Bayes":
                return dict()
                # return dict(clf__priors=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6],
                #                          [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])

            elif classifierName == "Logistic Regression":
                return dict(clf__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            clf__solver=["newton-cg", "lbfgs", "liblinear", "sag"])
            else:
                print("\nUnable to get classifier parameters!\n")
                return dict()

        def mergeDict(d1, d2):
                return dict(list(d1.items()) + list(d2.items()))

        for trainSet in trainSets:
            lista_statistic = list()
            # leitura da base para gridSearch
            a = CsvUtils.LoadData(trainSet, classe)
            X_train, y_train = a.splitDataFromClass()
            for clf, name in zip(clfs, names):
                for technique, t_name in zip(sTechniques, technique_names):
                    paramsRs = getParamsReSampling(technique)
                    paramsClf = getParamsClassifier(name)
                    if technique == None:
                        pipeline = Pipeline([('clf', clf)])
                    else:
                        pipeline = Pipeline([('smt', technique), ('clf', clf)])
                    dParams = mergeDict(paramsRs, paramsClf)
                    grid_search = GridSearchCV(pipeline, param_grid=[dParams], n_jobs=-1, scoring='roc_auc')
                    # BM
                    print("\n\nGrid search of", name, t_name, "using", trainSet, "\n\n")
                    grid_search.fit(X_train, y_train)
                    # print the classifier in order to show the parameters
                    print(grid_search.best_params_)
                    
                    # storing the classifier with the best parameters
                    classifier = pipeline.set_params(**grid_search.best_params_)
                    
                    dump_name = './models/best_' + row['train_names'] + '_' + name + '_' + t_name + '_non_trained.pkl'
                    
                    # dump the non-trained classifier
                    with open(dump_name, 'wb') as fid:
                        cPickle.dump(classifier, fid)
                    
                    # fit on the train data
                    classifier.fit(X_train.values, y_train.values)
                    
                    dump_name = './models/best_' + row['train_names'] + '_' + name + '_' + t_name + '_trained.pkl'
                    
                    # dump the trained lassifier
                    with open(dump_name, 'wb') as fid:
                        cPickle.dump(classifier, fid)
                    
                    # defino a lista de tecnicas de amostragem como None pois ja possuo a info completa do codigo anterior
                    # BM added the training set name
                    # BM now passing the fully trained classifier
                    
                    for test_set_name in testSets:
                        print("\n\nWill classify using", name, t_name, "on", test_set_name, "\n\n")
                        
                        print('./data/' + test_set_name+  '.csv')
                    
                        m = ClassifyCV(foldslist=None,
                                    dataset='./data/' + test_set_name+  '.csv',
                                    clf=classifier,
                                    clf_label=name,
                                    trainingSet=trainSet.split('/')[2].split('.')[0] + ' ' + './data/' + test_set_name+  '.csv',
                                    classe=classe,
                                    resamplingtechnique=None,
                                    techiqueLabel=t_name,
                                    tosql=False,
                                    verbose=True)
                        
                        m.classify()
                        stats_results = m.showStats()
                        stats_results['train_set'] = [trainSet] * len(stats_results.index)
                        stats_results['params'] = [grid_search.best_params_]
                        
                        lista_statistic.append(stats_results)

            df_final = pd.concat(lista_statistic, ignore_index=True)
            #save the csv for statistics
            #BM
            # from datetime import datetime
            all_results = pd.concat([all_results, df_final], axis=0)
            all_results.to_csv('results/baseline_static_smote_borderline_2.csv', columns=df_final.columns.values, index=False)
            # now = datetime.now() # current date and time
            # time_in_string = now.strftime("%H_%M_%S")
            # df_final.to_csv('results/' + row['train_names'] + '_' + row['test_names'] + '_baselines_rs_' + time_in_string + '.csv', columns=df_final.columns.values, index=False)
            
            print("Finished")
            print(trainSet)
            print(test_set_name)