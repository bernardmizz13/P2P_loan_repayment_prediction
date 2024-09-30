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
        clfs = [RandomForestClassifier()]
        names = ["Random Forest"]
        final_names = list()
        for set in testSets:
            for name in names:
                final_names.append(str(name+'_'+set[:-4]))

        # defino a lista de tecnicas de sampling
        sTechniques = [RandomUnderSampler(random_state=1)]
        technique_names = ["RU"]


        def getParamsReSampling(reSamplingTechnique):

            if type(reSamplingTechnique) is RandomUnderSampler:
                return dict(smt__ratio=[0.8, 0.9, 1.0])

            else:
                return dict()

        def getParamsClassifier(classifierName):
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
            all_results.to_csv('results/ensemble_static_random_undersampling.csv', columns=df_final.columns.values, index=False)
            # now = datetime.now() # current date and time
            # time_in_string = now.strftime("%H_%M_%S")
            # df_final.to_csv('results/' + row['train_names'] + '_' + row['test_names'] + '_baselines_rs_' + time_in_string + '.csv', columns=df_final.columns.values, index=False)
            
            print("Finished")
            print(trainSet)
            print(test_set_name)