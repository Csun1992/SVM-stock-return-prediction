import numpy as np
from sys import exit
from collections import Counter
from sklearn.decomposition import KernelPCA as pca
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn.metrics import f1_score
from classifier import Classifier

# Object that for svm with clustering 
class svmStockPred(Classifier):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        Classifier.__init__(self, microDataLoc, clusterNum, macroDataLoc="data/clusterData.txt")

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [svm.SVC() for i in range(self.clusterNum)]
        cvScore = []
        """
        kf = model_selection.KFold(n_splits = 40)
        totalErr = 0
        print kf.split(train[1])
        for trainIndex, testIndex in kf.split(train[1]):
            trainDat, testDat = train[1][trainIndex], train[1][testIndex]
            trainLabelcv, testLabelcv = trainLabel[1][trainIndex], trainLabel[1][testIndex]
            clf[1].fit(trainDat, trainLabelcv)
            result = clf[1].predict(testDat)
            length = len(result)
            print result
            print testLabelcv
            print '\n'
            error = sum([i != j for (i, j) in zip(result, testLabelcv)])
            totalErr = totalErr + error/float(length)
        print totalErr/20
        """
        for i in range(self.clusterNum):
            score = model_selection.cross_validate(clf[i], train[i], trainLabel[i], cv = 5, scoring
                 = 'f1', return_train_score = True) 
            cvScore.append(score['test_score'].mean()) 
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel, cvScore) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again


# StockPredNoClassification class is a class to classify the stock price direction 
# without using clustering. The macro data used for clustering now was combined with
# micro data for classification
class svmNoCluster(svmStockPred):
    def __init__(self, microDataLoc):
        svmStockPred.__init__(self, microDataLoc)

    def prepareData(self):
        group, label = [], []
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        labels = data[:, -1]
        nComponents = min(max(np.size(data, 0)/ 30, 1), np.size(data, 1)) # each feature needs 30 samples
        data = pca(n_components = nComponents, kernel = 'linear').fit_transform(data[:, :-1])
        for i in range(self.clusterNum):
            group.append(data)
            label.append(labels)
        return (group, label)

    def reportResult(self):
        f1, cv = self.test()
        print "Without Clustering, the f1 score is"
        print f1[0]
        print '\n'
        return f1 

            
if __name__ == "__main__":
    companies = ['microsoft', 'apple', 'att', 'sony', 'gap', 'fedex', 'mcdonalds', 'nike',
    'tiffany', 'homeDepot', 'walmart', 'cocaCola', 'avon', 'oracle', 'ibm', 'intel',
    'harley-davidson', 'toyota', 'honda', 'boeing', 'jpmorgan', 'boa', 'amgen', 'hermanMiller',
    'nissan', 'generalElectric', 'nextEra', 'conocoPhillips', 'bakerHughes', 'dukeEnergy', 'chevron']
    for companyName in companies:
        print "The stock we are considering now is " + companyName
        name = "data/" + companyName + "TrainData.txt"
        # without clustering
        stock = svmNoCluster(name)
        stock.reportResult()
        print '\n'
# # Case when 4 clusters
#        stock = svmStockPred(name, clusterNum=4)
#        stock.reportResult()
#   
         # for the case when cluster = 3
        stock = svmStockPred(name, clusterNum = 3)
        stock.reportResult()
#
#        # Case when 2 clusters
#        stock = svmStockPred(name, clusterNum=2)
#        stock.reportResult()
#
#        # Case when 1 cluster
#        stock = svmStockPred(name, clusterNum=1)
#        stock.reportResult()
#        print '\n'
