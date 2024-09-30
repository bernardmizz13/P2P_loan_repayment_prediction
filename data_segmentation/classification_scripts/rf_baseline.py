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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

import _pickle as cPickle
import numpy as np

#BM
NUM_PARALLEL_EXEC_UNITS = 6

def getParamsClassifierEnsamble(classifierName):
    # Importante: eu acho que soh usa max_depth quando max_leaf_nodes = None e vice-versa. Portanto nao aumenta muito
    # a quantidade de modelos a testar quando adicionamos mais valores para ambos (como eu fiz :P)
    if classifierName == "Random Forest":
        return dict(clf__n_estimators=[10, 50, 100],  # Good
                    # clf__criterion=["gini", "entropy"])#, # Good
                    clf__max_features=["auto", "sqrt", "log2", None],  # Esta ok, poderia ter mais valores
                    clf__max_depth=[5, 10,
                                    15])  # retirei o none para evitar testar com arvores sem limite de tamanho
        # clf__min_samples_split=[10, 50, 100], # Retirado por questoes de performance
        # clf__min_samples_leaf=[10, 50, 100, 200]) # Retirado por questoes de performance
        # clf__max_leaf_nodes=[None, 20, 50, 100], # Tirei esse. Ele eh mutualmente exclusivo com o max_depth e normalmente otimizamos o max_depth (eh mais compreensivel)
        # clf__bootstrap=[True, False]) # Good

    elif classifierName == "AdaBoost":
        return dict(clf__n_estimators=[10, 50, 100],  # Good, alinhado com RF
                    clf__learning_rate=[0.1, 1, 2],  # Mudei para 0.1, 1 e 2
                    clf__algorithm=["SAMME", "SAMME.R"])

    elif classifierName == "Bagging":
        return dict(clf__n_estimators=[10, 50, 100],  # Alinhado com os outros
                    clf__max_samples=[0.10, 0.25, 0.5, 0.75, 1.0],
                    # Coloquei mais um valor baixo (10%), provavelmente nao vai melhorar o resultado final, mas soh pra fins de teste mesmo.
                    clf__max_features=[0.10, 0.25, 0.5, 0.75,
                                       1.0])  # Mesma coisa, coloquei um outro valor baixo. Aqui tem mais chances the melhorar o resultado final, pois temos muitas features.
        # clf__bootstrap=[True, False], # Ok
        # clf__bootstrap_features=[True, False]) # Ok
        # clf__warm_start=[True, False]) # Nao se aplica a grid search, seria reutilizar o modelo utilizado na chamada anterior ao fit
    else:
        print("\nUnable to get classifier parameters!\n")
        return dict()


def mergeDict(d1, d2):
    return dict(list(d1.items()) + list(d2.items()))


def generate_names(dbases, labels):
    final_names = list()
    for set in dbases:
        for name in labels:
            final_names.append(str(name + '_' + set[:-4]))
    return final_names


if __name__ == '__main__':

    # BM read the training and testing set names
    info = pd.read_csv('./data/info.csv')
    
    all_results = pd.DataFrame()
    
    # BM for each training and testing set
    for index, row in info.iterrows():
        
        # BM
        trainSets = ['./data/' + row['train_names'] + '.csv']
        testSets = ['./data/' + row['test_names'] + '.csv']

        classe = 'loan_status'
        clfs = [RandomForestClassifier()]
        names = ["Random Forest"]
        sTechniques = [None]
        technique_names = ["Ensemble-based"]
        final_names = generate_names(testSets, names)
        
        for trainSet, testSet, name in zip(trainSets, testSets, final_names):
            lista_statistic = list()
            # leitura da base para gridSearch
            a = CsvUtils.LoadData(trainSet, classe)
            X_train, y_train = a.splitDataFromClass()
            for clf, name in zip(clfs, names):
                paramsClf = getParamsClassifierEnsamble(name)
                pipeline = Pipeline([('clf', clf)])
                grid_search = GridSearchCV(pipeline, param_grid=[paramsClf], n_jobs=5, scoring='roc_auc')
                # BM
                print("\n\nGrid search of", name, 'baseline', "using", trainSet, "\n\n")
                grid_search.fit(X_train, y_train)
                # print the classifier in order to show the parameters
                print(grid_search.best_params_)
                
                # storing the classifier with the best parameters
                classifier = pipeline.set_params(**grid_search.best_params_)
                
                dump_name = './models/best_' + row['train_names'] + '_' + name + '_' + 'baseline' + '_non_trained.pkl'
                
                # dump the non-trained classifier
                with open(dump_name, 'wb') as fid:
                    cPickle.dump(classifier, fid)
                
                # fit on the train data
                classifier.fit(X_train.values, y_train.values)
                
                dump_name = './models/best_' + row['train_names'] + '_' + name + '_' + 'baseline' + '_trained.pkl'
                
                # dump the trained lassifier
                with open(dump_name, 'wb') as fid:
                    cPickle.dump(classifier, fid)
                
                # defino a lista de tecnicas de amostragem como None pois ja possuo a info completa do codigo anterior
                # BM added the training set name
                # BM now passing the fully trained classifier
                print("\n\nWill classify using", name, 'baseline', "on", testSet, "\n\n")
                m = ClassifyCV(foldslist=None,
                            dataset=testSet,
                            clf=classifier,
                            clf_label=name,
                            trainingSet=trainSet.split('/')[2].split('.')[0] + ' ' + testSet,
                            classe=classe,
                            resamplingtechnique=None,
                            techiqueLabel=name,
                            tosql=False,
                            verbose=True)
                
                m.classify()
                stats_results = m.showStats()
                stats_results['train_set'] = [trainSet] * len(stats_results.index)
                    
                lista_statistic.append(stats_results)

            df_final = pd.concat(lista_statistic, ignore_index=True)
            #save the csv for statistics
            #BM
            # from datetime import datetime
            all_results = pd.concat([all_results, df_final], axis=0)
            all_results.to_csv('results/ensemble.csv', columns=df_final.columns.values, index=False)
            # now = datetime.now() # current date and time
            # time_in_string = now.strftime("%H_%M_%S")
            # df_final.to_csv('results/' + row['train_names'] + '_' + row['test_names'] + '_baselines_rs_' + time_in_string + '.csv', columns=df_final.columns.values, index=False)
            
            print("Finished")
            print(trainSet)
            print(testSet)