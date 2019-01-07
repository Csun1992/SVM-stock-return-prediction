import numpy as np
from sys import exit
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import KernelPCA as pca
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from classifier import Classifier


def f1Score(precision, recall):
    return 1.0/(1.0 / precision + 1.0 / recall)

def plotPrecisionRecall(precision, recall, thresholds):
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g--", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    


# Object that for svm with clustering 
class svmStockPred(Classifier):
    def __init__(self, microDataLoc, macroDataLoc="data/clusterData.txt"):
        Classifier.__init__(self, microDataLoc, macroDataLoc="data/clusterData.txt")

    def train(self):
        cvForDiffClusters = []
        for clusterNum in range(1, 5):
            train, test, trainLabel, testLabel = self.trainTestSplit(clusterNum)
            clf = svm.SVC(C=1000, kernel='rbf') 
            print sum(trainLabel[0])/float(len(trainLabel[0]))
            yScores = model_selection.cross_val_predict(clf, train[0], trainLabel[0], cv=3,
                    method='decision_function')
            precision, recall, thresholds = precision_recall_curve(trainLabel[0], yScores)
#  print precision
#            print recall
#            print thresholds
            plotPrecisionRecall(precision, recall, thresholds)
            plt.show()
            plt.plot(precision[:-1], recall[:-1], 'b--', label='precision_vs_recall')
            plt.xlabel("precision")
            plt.ylabel("recall")
            plt.show()
        exit()
    """
            cvScore = []
            clf = svm.SVC(C=1, kernel='rbf') 
            for i in range(clusterNum):
                score = model_selection.cross_validate(clf, train[i], trainLabel[i], cv = 5, scoring
                     = 'precision', return_train_score = True) 
                cvScore.append(score['test_score'].mean()) 
            cvForDiffClusters.append(sum(cvScore)/float(len(cvScore)))
        print cvForDiffClusters
        clusterNum = cvForDiffClusters.index(max(cvForDiffClusters)) + 1 
        clf = [svm.SVC(C=1, kernel='rbf') for i in range(clusterNum)]
        for i in range(clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel, clusterNum) # return test and testLabel to self.test() so no need to
    """                                 # recompute the testing data again


# StockPredNoClassification class is a class to classify the stock price direction 
# without using clustering. The macro data used for clustering now was combined with
# micro data for classification
class svmNoCluster(svmStockPred):
    def __init__(self, microDataLoc):
        svmStockPred.__init__(self, microDataLoc)

    def prepareData(self):
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        label = data[:, -1]
        data = data[:, :-1]
        """
        nComponents = min(max(np.size(data, 0)/ 30, 1), np.size(data, 1)) # each feature needs 30 samples
        data = pca(n_components = nComponents, kernel = 'linear').fit_transform(data[:, :-1])
        """
        return (data, label)

    def trainTestSplit(self):
        data, label = self.prepareData()
        train, test, trainLabel, testLabel= model_selection.train_test_split(data, label, test_size=0.3, random_state=11)
        return (train, test, trainLabel, testLabel)
        
    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = svm.SVC(C=100, kernel='poly', degree=3) 
        clf.fit(train, trainLabel)
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to

    def test(self):
        clf, test, testLabel = self.train()
        pred = clf.predict(test) 
        f1 = precision_score(testLabel, pred)
        return f1

    def reportResult(self):
        f1 = self.test()
        print "Without Clustering, the f1 score is"
        print f1
        print '\n'
        return f1 

            
if __name__ == "__main__":
#    companies = ['microsoft', 'apple', 'att', 'sony', 'gap', 'fedex', 'mcdonalds', 'nike',
#    'tiffany', 'homeDepot', 'walmart', 'cocaCola', 'avon', 'oracle', 'ibm', 'intel',
#    'harley-davidson', 'toyota', 'honda', 'boeing', 'jpmorgan', 'boa', 'amgen', 'hermanMiller',
#    'nissan', 'generalElectric', 'nextEra', 'conocoPhillips', 'bakerHughes', 'dukeEnergy', 'chevron']
    companies = ['mcdonalds']
    for companyName in companies:
        print "The stock we are considering now is " + companyName
        name = "data/" + companyName + "TrainData.txt"
        # without clustering
        stock = svmNoCluster(name)
        stock.reportResult()
#        print '\n'
# # Case when 4 clusters
#        stock = svmStockPred(name, clusterNum=4)
#        stock.reportResult()
#   
         # for the case when cluster = 3
        stock = svmStockPred(name)
        stock.reportResult()
#        # Case when 2 clusters
#        stock = svmStockPred(name, clusterNum=2)
#        stock.reportResult()
#
#        # Case when 1 cluster
#        stock = svmStockPred(name, clusterNum=1)
#        stock.reportResult()
#        print '\n'
