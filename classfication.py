import numpy as np
from sklearn import preprocessing, cluster, model_selection, svm
from sys import exit

# Object that is for svm with classification
class StockPrediction(object):
    def __init__(self, microDataLoc, clusterNum=3, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc
        self.clusterNum = clusterNum

    def cluster(self):
        data = np.loadtxt(self.macroDataLoc)
        cleanData = preprocessing.scale(data)
        self.kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(self.kmeans.labels_)
        return groupNum

    def prepareData(self):
        self.group, self.label = [], []
        groupNum = self.cluster()
        data = np.loadtxt(self.microDataLoc)
        for i in range(self.clusterNum):
            self.group.append(data[groupNum==i, :-1])
            self.label.append(data[groupNum==i, -1])
        return (self.group, self.label)

    def trainTestSplit(self):
        self.train, self.trainLabel = [], []
        self.test, self.testLabel = [], []
        for i in range(self.clusterNum):
            train, test, trainLabel, testLabel = model_selection.train_test_split(self.group[i],
                    self.label[i], test_size=0.3, random_state=11)
            self.train.append(train)
            self.test.append(test)
            self.trainLabel.append(trainLabel)
            self.testLabel.append(testLabel)
        return (self.train, self.test, self.trainLabel, self.testLabel)
            
    def train(self):
        self.prepareData()
        self.trainTestSplit()
        self.clf = [svm.SVC() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            self.clf[i].fit(self.train[i], self.trainLabel[i])
        return self.clf

    def test(self):
        self.train()
        self.error = []
        for i in range(self.clusterNum):
            pred = (self.clf[i].predict(self.test[i]) == 1)
            error = sum([i != j for (i,j) in zip(self.testLabel[i], pred)])
            self.error.append(float(error)/len(pred))
        return self.error

    def reportResult(self):
        self.test()
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " correct rate"
            print 1-self.error[i]
        print '\n'
        return self.error


class StockPredNoClassification(StockPrediction):
    def __init__(self, microDataLoc, macroDataLoc="data/clusterData.txt"):
        StockPrediction.__init__(self, microDataLoc, clusterNum=1, macroDataLoc=macroDataLoc)

    def prepareData(self):
        self.group, self.label = [], []
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        for i in range(self.clusterNum):
            self.group.append(data[: , :-1])
            self.label.append(data[: , -1])
        return (self.group, self.label)

    def reportResult(self):
        self.test()
        print "Without Clustering, the correct classification rate is"
        print 1-self.error[0]
        print '\n'
        return self.error

            
if __name__ == "__main__":

    # without clustering
    apple = StockPredNoClassification("data/appleTrainData.txt")
    apple.reportResult()

    # for the case when cluster = 3
    apple = StockPrediction("data/appleTrainData.txt")
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
    att = StockPrediction("data/attTrainData.txt")
    att.reportResult()
        
    # Case when 2 clusters
    att = StockPrediction("data/attTrainData.txt", clusterNum=2)
    att.reportResult()
       
    # Case when 4 clusters
    att = StockPrediction("data/attTrainData.txt", clusterNum=4)
    att.reportResult()
