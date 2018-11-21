import numpy as np
from sklearn import preprocessing, cluster, model_selection, svm

# Object that for svm with clustering 
class StockPrediction(object):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc
        self.clusterNum = clusterNum

    def cluster(self):
        data = np.loadtxt(self.macroDataLoc)
        cleanData = preprocessing.scale(data)
        kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(kmeans.labels_)
        return groupNum

    def prepareData(self):
        data = np.loadtxt(self.microDataLoc)
        groupNum = self.cluster()
        group, label = [], []
        for i in range(self.clusterNum):
            group.append(data[groupNum==i, :-1])
            label.append(data[groupNum==i, -1])
        return (group, label)

    def trainTestSplit(self):
        train, test, trainLabel, testLabel = [], [], [], []
        group, label = self.prepareData()
        for i in range(self.clusterNum):
            trainData, testData, trainLabelData, testLabelData = model_selection.train_test_split(group[i],
                    label[i], test_size=0.3, random_state=11)
            train.append(trainData)
            test.append(testData)
            trainLabel.append(trainLabelData)
            testLabel.append(testLabelData)
        return (train, test, trainLabel, testLabel)
            
    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [svm.SVC() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again

    def test(self):
        clf, test, testLabel = self.train()
        error = []
        for i in range(self.clusterNum):
            pred = (clf[i].predict(test[i]) == 1)
            caseError = sum([i != j for (i,j) in zip(testLabel[i], pred)])
            error.append(float(caseError)/len(pred))
        return error

    def reportResult(self):
        error = self.test()
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " correct rate"
            print 1-error[i]
        print '\n'
        return error


# StockPredNoClassification class is a class to classify the stock price direction 
# without using clustering. The macro data used for clustering now was combined with
# micro data for classification
class StockPredNoClassification(StockPrediction):
    def __init__(self, microDataLoc):
        StockPrediction.__init__(self, microDataLoc)

    def prepareData(self):
        group, label = [], []
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        for i in range(self.clusterNum):
            group.append(data[: , :-1])
            label.append(data[: , -1])
        return (group, label)

    def reportResult(self):
        error = self.test()
        print "Without Clustering, the correct classification rate is"
        print 1-error[0]
        print '\n'
        return error

            
if __name__ == "__main__":

    # without clustering
    apple = StockPredNoClassification("data/appleTrainData.txt")
    apple.reportResult()

    # for the case when cluster = 3
    apple = StockPrediction("data/appleTrainData.txt", clusterNum=3)
    apple.reportResult()
        
    # Case when 2 clusters
    apple = StockPrediction("data/appleTrainData.txt", clusterNum=2)
    apple.reportResult()
       
    # Case when 4 clusters
    apple = StockPrediction("data/appleTrainData.txt", clusterNum=4)
    apple.reportResult()

    # Case when 1 cluster
    apple = StockPrediction("data/appleTrainData.txt", clusterNum=1)
    apple.reportResult()


    
    # without clustering
    att = StockPredNoClassification("data/attTrainData.txt")
    att.reportResult()

    # for the case when cluster = 3
    att = StockPrediction("data/attTrainData.txt", clusterNum=3)
    att.reportResult()
        
    # Case when 2 clusters
    att = StockPrediction("data/attTrainData.txt", clusterNum=2)
    att.reportResult()
       
    # Case when 4 clusters
    att = StockPrediction("data/attTrainData.txt", clusterNum=4)
    att.reportResult()

    # Case when 1 cluster
    att = StockPrediction("data/attTrainData.txt", clusterNum=1)
    att.reportResult()
