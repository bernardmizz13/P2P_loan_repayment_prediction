from sklearn.metrics import confusion_matrix, roc_curve, auc
import time
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import math
import sqlite3
import pandas as pd
# BM for auc roc plotting
import matplotlib.pyplot as plt

from datetime import datetime

now = datetime.now() # current date and time
time_in_string = now.strftime("%H_%M_%S")

# BM
import os
import logging

#BM
NUM_PARALLEL_EXEC_UNITS = 6

# initializing the logger
logging.basicConfig(filename=os.path.realpath("results/classification_results_") + time_in_string + '.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logging.info("Initialized logger")
print("Initialized logger")

class ClassifyCV:
    """Esta classe é recebe 6 parâmetros:
    -> foldslist (list) eh a lista de combinações dos folds gerados pela classe CrossValidationStratified
    -> clf (método) é o classificador utilizado (único)
    -> classe (string) é a classe (deerrr), se não for passado nenhum parâmetro a ultima coluna do
    dataset é utilizada
    -> resamplingTechnique (string) é a técnica de over / under sampling aplicada. Se não informado
    nenhuma técnica sera utilizada.
    -> toSql (bool) responsável por salvar os resultados ou não em um sqlite previamente criado
    ->verbose (bool) é referente a impressão de alguns valores na tela durante a execução.
    Classe criada por Luis Eduardo Boiko ferreira
    luiseduardo.boiko@gmail.com
    ultima atualização: 22/11/2016"""

    # BM added trainingSet
    def __init__(self, foldslist, dataset, clf, clf_label, trainingSet, scl, cls_t_sc, classe="lastCol", resamplingtechnique=None,
                 techiqueLabel=None, tosql=False, verbose=False):
        self.foldsList = foldslist
        self.dataset = dataset
        self.clf = clf
        self.clf_label = clf_label
        self.classe = classe
        self.resamplingTechnique = resamplingtechnique
        self.techniqueLabel = techiqueLabel
        self.toSql = tosql
        self.verbose = verbose
        # BM
        self.trainingSet = trainingSet
        
        # BM get the name of the training set
        self.testSetName = trainingSet.split()[1]
        
        # for the scaler
        self.scl = scl
        self.cls_t_sc = cls_t_sc
        
        # criando as listas
        self.listaCM = list()
        self.listaAcuracia = list()
        self.listaPrecision = list()
        self.listaRecall = list()
        self.listaAUC = list()
        self.listaAUCProb = list()
        self.listaMcc = list()
        self.listaF1 = list()
        # variaveis para as bases
        self.treino_X, self.treino_Y, self.teste_X, self.teste_Y = None, None, None, None
        self.y_pred = None
        self.y_pred_prob = None

        # variaveis para as metricas
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.accuracy, self.precisionMin, self.precisionMaj, self.recallMin, self.recallMaj = 0, 0, 0, 0, 0
        self.f1Min, self.f1Maj, self.gMin, self.gMaj, self.aucRes, self.aucResProb, self.mcc = 0, 0, 0, 0, 0, 0, 0

        # variaveis para estatistica
        self.statisticsList = list()

        # separador
        self.separator = 45 * '-'
    
    def compute_my_auc_roc(self, orig_labels, predict_probs):
        
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
        
        # check name
        training_set_name = ''
        test_set_name = ''
        
        if len(self.trainingSet.split()) > 1:
            test_set_name = self.trainingSet.split('/')[2].split('.')[0]
        if len(self.trainingSet.split()) > 1:
            training_set_name = self.trainingSet.split('/')[0][:len(self.trainingSet.split('/')[0])-2]
        
        plt.savefig('./results/' + self.clf_label + '_' + self.techniqueLabel + '_' + training_set_name + '_' + test_set_name + '_my_auc_roc_prob.png')
        plt.close()
        
        return auc
        
    
    def classify(self):
        """
        Este método é responsável por realizar a classificação das instâncias.
        Utiliza para isto a lista de folds combinadas geradas pela classe CrossValidationStratified.
        """
        if self.classe == "lastCol":
            self.classe = list(self.foldsList[0][0].columns.values)[-1]
        
        startCl = time.time()
        
        # BM read the test data
        print('\n\nTrained on and will classify (respectively):', self.trainingSet, '\n\n')
        logging.info('\n\nTrained on and will classify (respectively): ' + self.trainingSet + '\n\n')
        logging.info("\n\nReading " + self.dataset + " to classify it\n\n")
        print("\n\nReading", self.dataset, "to classify it\n\n")
        test_data = pd.read_csv(self.dataset)
        
        print(test_data.head(5))
        
        # drop the loan status column (class label)
        self.teste_X = test_data.drop(self.classe, axis=1)
        self.teste_Y = test_data[self.classe]
        
        # BM for scaling
        self.teste_X[self.cls_t_sc] = self.scl.transform(self.teste_X[self.cls_t_sc])
        
        # BM moved testing on the outside
        
        # BM check if XGBoost since to avoid feature name errors
        if self.clf_label == 'XGBoost':
            self.teste_X = self.teste_X.values
        
        # BM, do not fit the model again to not lose weights
        self.y_pred = self.clf.predict(self.teste_X)
        # get the probabilities that a loan is defaulted
        self.y_pred_prob = self.clf.predict_proba(self.teste_X)[:,1]
        
        # check name
        training_set_name = ''
        test_set_name = ''
        
        print("\n\n", self.trainingSet, "\n\n")
        
        if len(self.trainingSet.split()) > 1:
            test_set_name = self.trainingSet.split('/')[2].split('.')[0]
        if len(self.trainingSet.split()) > 1:
            training_set_name = self.trainingSet.split('/')[0][:len(self.trainingSet.split('/')[0])-2]
        
        preds_df = pd.DataFrame({'test': self.teste_Y, 'pred_labels': self.y_pred.ravel(), 'pred_prob': self.y_pred_prob})
        preds_df.to_csv('./predictions/' + training_set_name + '_' + test_set_name + '_' + self.clf_label + '_' + self.techniqueLabel + '_predictions.csv', index = False)
        
        # BM now passing the fold number
        self.addmetric()
        
        if self.verbose:
            print(self.separator)
        self.showresults()

    # BM now receiving the fold number
    def addmetric(self):
        """
        Este método adiciona os valores de cada fold a lista referente a cada metrica
        """
        # Bm now ravel on cnfusion matrix
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.teste_Y, self.y_pred).ravel()
        
        # BM save the confusion matrix
        test_conf_matrix = confusion_matrix(self.teste_Y, self.y_pred)
        
        # check name
        training_set_name = ''
        test_set_name = ''
        if len(self.trainingSet.split()) > 1:
            test_set_name = self.trainingSet.split('/')[2].split('.')[0]
        if len(self.trainingSet.split()) > 1:
            training_set_name = self.trainingSet.split('/')[0][:len(self.trainingSet.split('/')[0])-2]
        
        f = open('./confusion_matrix/' + training_set_name + '_' + test_set_name + '_' + self.clf_label + '_' + self.techniqueLabel + '_confusion_matrix.csv', 'w')
        f.write('\n\nConfusion Matrix\n\n{}\n'.format(test_conf_matrix))
        f.close()
        
        self.listaAcuracia.append(accuracy_score(self.teste_Y, self.y_pred))
        self.listaRecall.append(recall_score(self.teste_Y, self.y_pred, average=None))
        self.listaPrecision.append(precision_score(self.teste_Y, self.y_pred, average=None))
        self.listaF1.append(f1_score(self.teste_Y, self.y_pred, average=None))
        # auc
        # Compute ROC curve and ROC area
        
        
        fpr, tpr, _ = [], [], []
        fpr_prob, tpr_prob, _prob = [], [], []
        
        fpr, tpr, _ = roc_curve(self.teste_Y.ravel(), self.y_pred.ravel())
        fpr_prob, tpr_prob, _prob = roc_curve(self.teste_Y.ravel(), self.y_pred_prob)
            
        
        
        # BM plotting AUC ROC of each fold
        plt.figure()
        plt.plot(fpr, tpr, '--r') # dashed red
        plt.plot([0, 1], [0, 1], '-b') # straigt line blue
        plt.title('AUC ROC')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        # check name
        training_set_name = ''
        test_set_name = ''
        
        if len(self.trainingSet.split()) > 1:
            test_set_name = self.trainingSet.split('/')[2].split('.')[0]
        if len(self.trainingSet.split()) > 1:
            training_set_name = self.trainingSet.split('/')[0][:len(self.trainingSet.split('/')[0])-2]
            
        plt.savefig('./results/' + self.clf_label + '_' + self.techniqueLabel + '_' + training_set_name + '_' + test_set_name + '_binary.png')
        plt.close()
        
        # BM plotting AUC ROC of each fold
        plt.figure()
        plt.plot(fpr_prob, tpr_prob, '--r') # dashed red
        plt.plot([0, 1], [0, 1], '-b') # straigt line blue
        plt.title('AUC ROC')
        # close the figure
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.savefig('./results/' + self.clf_label + '_' + self.techniqueLabel + '_' + training_set_name + '_' + test_set_name + '_prob.png')
        plt.close()
        
        # BM get the auc roc of the current fold
        fold_auc_roc_prob = auc(fpr_prob, tpr_prob)
        fold_auc_roc = auc(fpr, tpr)
        print("False positive rate")
        print(fpr)
        print("True positive rate")
        print(tpr)
        print("Threshold")
        print(_)
        
        logging.info("False positive rate")
        logging.info(fpr)
        logging.info("True positive rate")
        logging.info(tpr)
        logging.info("Treshold")
        logging.info(_)
                
        # BM printing AUC ROC result of each fold
        print('Binary AUC:', str(fold_auc_roc))
        logging.info('Binary AUC: ' + str(fold_auc_roc))
        
        print('Prob AUC:', str(fold_auc_roc_prob))
        logging.info('Prob AUC: ' + str(fold_auc_roc_prob))
        
        # fpr, tpr, thresholds = roc_curve(self.teste_Y, self.y_pred, pos_label=1)
        self.listaAUC.append(fold_auc_roc)
        self.listaAUCProb.append(fold_auc_roc_prob)


    def showresults(self):
        """
        Este método apenas mostra os resultados, fazendo as médias dos valores
        """
        # matriz de confusão
        # BM amended the denominators
        self.accuracy = self.listaAcuracia[0]
        self.precisionMaj = [sum(i) for i in zip(*self.listaPrecision)][0] / len(self.listaPrecision)
        self.precisionMin = [sum(i) for i in zip(*self.listaPrecision)][1] / len(self.listaPrecision)
        self.recallMaj = [sum(i) for i in zip(*self.listaRecall)][0] / len(self.listaRecall)
        self.recallMin = [sum(i) for i in zip(*self.listaRecall)][1] / len(self.listaRecall)
        self.f1Maj = [sum(i) for i in zip(*self.listaF1)][0] / len(self.listaF1)
        self.f1Min = [sum(i) for i in zip(*self.listaF1)][1] / len(self.listaF1)

        self.gMaj = math.sqrt(self.precisionMaj * self.recallMaj)
        self.gMin = math.sqrt(self.precisionMin * self.recallMin)
        # mcc
        # self.mcc = (self.tp * self.tn - (self.fp - self.fn)) / math.sqrt((self.tp + self.fp) *
        #                                                                 (self.tp + self.fn) *
        #                                                                 (self.tn + self.fp) *
        #                                                                 (self.tn + self.fn))
        # auc
        # BM changed since only one value
        self.aucRes = self.listaAUC[0]
        self.aucResProb = self.listaAUCProb[0]
        # statistic
        colunas =['DataSet', 'Classifier', 'AUCROC_label', 'AUCROC_prob', 'MY_AUC_ROC_prob', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'TPs', 'FPs', 'TNs', 'FNs']
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        
        tns, fps, fns, tps = confusion_matrix(self.teste_Y, self.y_pred).ravel()
        
        # BM compute my auc        
        my_auc = self.compute_my_auc_roc(orig_labels = self.teste_Y.ravel(), predict_probs = self.y_pred_prob)
        
        self.statisticsList.append(pd.DataFrame(dict(DataSet=self.dataset,
                                                    Classifier=self.clf_label + '+' + self.techniqueLabel,
                                                    AUCROC_label=self.listaAUC, AUCROC_prob=self.listaAUCProb, MY_AUC_ROC_prob = my_auc,
                                                    Accuracy=accuracy_score(self.teste_Y.ravel(), self.y_pred.ravel()),
                                                    Precision=precision_score(self.teste_Y.ravel(), self.y_pred.ravel()),
                                                    Recall=recall_score(self.teste_Y.ravel(), self.y_pred.ravel()),
                                                    F1_score=f1_score(self.teste_Y.ravel(), self.y_pred.ravel()),
                                                    TPs = tps,
                                                    FPs = fps,
                                                    TNs = tns,
                                                    FNs = fns),
                                                    columns=colunas))

        #BM
        logging.info(self.clf_label + ' + ' + self.techniqueLabel)
        logging.info(self.clf.named_steps)
        logging.info('Training set ' + self.trainingSet)

        print('Class: {0:10d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
        
        logging.info('Class: {0:10d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
        
        print('Precision:   {0:10s} {1:10s}'.format('%.2f' % self.precisionMaj, '%.2f' % self.precisionMin))
        
        logging.info('Precision:   {0:10s} {1:10s}'.format('%.2f' % self.precisionMaj, '%.2f' % self.precisionMin))
            
        # print('Precision Min:   {0:2f}'.format(self.precisionMin))
        print('Recall:      {0:10s} {1:10s}'.format('%.2f' %  self.recallMaj, '%.2f' % self.recallMin))
            
        logging.info('Recall:      {0:10s} {1:10s}'.format('%.2f' %  self.recallMaj, '%.2f' % self.recallMin))
            
        # print('Recall Min:      {0:2f}'.format(self.recallMin))
        print('F1:          {0:10s} {1:10s}'.format('%.2f' %  self.f1Maj, '%.2f' % self.f1Min))
            
        logging.info('F1:          {0:10s} {1:10s}'.format('%.2f' %  self.f1Maj, '%.2f' % self.f1Min))
            
        # print('F1 Min:          {0:2f}'.format(self.f1Min))
        print('G-measure:   {0:10s} {1:10s}'.format('%.2f' %  self.gMaj, '%.2f' % self.gMin))
            
        logging.info('G-measure:   {0:10s} {1:10s}'.format('%.2f' %  self.gMaj, '%.2f' % self.gMin))
            
        # print('G-measure Min:   {0:2f}'.format(self.gMin))
        print('\nGlobal metrics:')
        print(self.separator)
        print('Acurácia:    {0:10s}'.format('%.2f' % self.accuracy))
            
        logging.info('Acurácia:    {0:10s}'.format('%.2f' % self.accuracy))
            
        print('MCC:         {0:10s}'.format('%.2f' % self.mcc))
            
        logging.info('MCC:         {0:10s}'.format('%.2f' % self.mcc))
        
        # BM changed to 6 DP    
        print('AUC binary labels:         {0:10s}'.format('%.6f' % self.aucRes))
            
        logging.info('AUC binary labels:         {0:10s}'.format('%.6f' % self.aucRes))
        
        print('AUC prob:         {0:10s}'.format('%.6f' % self.aucResProb))
            
        logging.info('AUC prob:         {0:10s}'.format('%.6f' % self.aucResProb))
            
        print('\nConfusion Matrix:')
            
        logging.info('\nConfusion Matrix:')
            
        print(self.separator)
        print('Class:  {0:3d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
            
        logging.info('Class:  {0:3d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
            
        print('{0}{1:10d} {2:10d}'.format([int(el) for el in self.teste_Y.unique()][0],
                                          int(round(self.tn)), int(round(self.fp))))
            
        logging.info('{0}{1:10d} {2:10d}'.format([int(el) for el in self.teste_Y.unique()][0],
                                          int(round(self.tn)), int(round(self.fp))))
            
        print('{0}{1:10d} {2:10d}\n\n'.format([int(el) for el in self.teste_Y.unique()][1],
                                          int(round(self.fn)), int(round(self.tp))))
            
        logging.info('{0}{1:10d} {2:10d}\n\n'.format([int(el) for el in self.teste_Y.unique()][1],
                                          int(round(self.fn)), int(round(self.tp))))
            
        # print('[[{0} {1:4}] \n [{2:4} {3:4}]]'.format(int(round(self.tp)),
        #       int(round(self.fp)), int(round(self.fn)), int(round(self.tn))))

        if self.toSql:
            # sqlite
            db = "resultados.sqlite"
            conn = sqlite3.connect(db)
            c = conn.cursor()
            c.execute("INSERT INTO individualBancos(base, tratamento, instanciasA, instanciasB,"
                      "acuracia, precisionMaj, precisionMin, recallMaj, recallMin,"
                      "f1Maj, f1Min, gMaj, gMin, mcc, auc, tp, tn, fp, fn, clf,imbLevel)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )", (
                          'lendingClub', str(self.resamplingTechnique), 666,
                          666, self.accuracy, self.precisionMaj,
                          self.precisionMin, self.recallMaj, self.recallMin, self.f1Maj, self.f1Min,
                          self.gMaj, self.gMin, self.mcc, self.aucRes,
                          int(round(self.tp)), int(round(self.tn)), int(round(self.fp)),
                          int(round(self.fn)), str(self.clf), str("1:4")))
            conn.commit()
            conn.close()

    def showStats(self):
        return pd.concat(self.statisticsList, ignore_index=True)