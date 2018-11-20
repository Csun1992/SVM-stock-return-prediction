import numpy as np
from sklearn import preprocessing, cluster, model_selection, svm

class stockPrediction:
    def __init__(self, stockDataLoc, clusterNum=3, fold=10, clusterDataLoc="data/clusterData.txt"):
        self.stockDataLoc = stockDataLoc
        self.clusterNum = clusterNum
        self.clusterDataLoc = clusterDataLoc
        self.fold = fold
        self.clusterStockPrice()
        self.train()
        self.crossValidation()
        self.findBaseRate()
        num,totalLen  = sum(map(sum, self.label)),sum(map(len, self.label))
        self.noClusteringBaseRate = max(float(num)/totalLen, 1-float(num)/totalLen) 

    def cluster(self):
        data = np.loadtxt(self.clusterDataLoc)
        cleanData = preprocessing.scale(data)
        self.kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(self.kmeans.labels_)
        return groupNum

    def clusterStockPrice(self):
        self.group, self.label = [], []
        groupNum = self.cluster()
        data = np.loadtxt(self.stockDataLoc)
        for i in range(self.clusterNum):
            self.group.append(data[groupNum==i, :-1])
            self.label.append(data[groupNum==i, -1])
        return (self.group, self.label)

    def trainTestSplit(self):
        self.train, self.trainLabel = [], []
        self.test, self.testLabel = [], []
        for i in range(self.clusterNum):
            train, test, trainLabel, testLabel = model_selection.train_test_split(self.group[i],
                    self.label[i], test_size=0.2, random_state=11)
            self.train.append(train)
            self.test.append(test)
            self.trainLabel.append(trainLabel)
            self.testLabeltrain.append(train)
        return (self.train, self.test, self.trainLabel, self.testLabel)
            
    def crossValidation(self):
        self.scores = []
        for i in range(self.clusterNum):
            self.scores.append(model_selection.cross_val_score(self.clf[i], self.group[i], self.label[i],
                    cv=self.fold))
        return self.scores 
    
    def train(self):
        self.clf = [svm.SVC() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            self.clf[i].fit(self.group[i], self.label[i])
        return self.clf

    def findBaseRate(self):
        self.baseRate = []
        for i in range(self.clusterNum):
            br = sum(self.label[i])/np.size(self.label[i], 0)
            br = max(br, 1-br)
            self.baseRate.append(br)
        return self.baseRate
    
    def reportResult(self):
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " correct rate"
            print self.scores[i].mean()
            print "Base Rate:"
            print self.baseRate[i]
            print "\n"

        print "And finally, if we do not have any clustering, the precision rate is:"
        print self.noClusteringBaseRate
        return self.scores

    def predict(self, macro, idiosync):
       pass 
   




            
if __name__ == "__main__":

    # for the case when cluster = 3
    apple = stockPrediction("data/appleTrainData.txt")
    apple.reportResult()
        
    # Case when 2 clusters
    apple = stockPrediction("data/appleTrainData.txt", clusterNum=2)
    apple.reportResult()
       
    # Case when 4 clusters
    apple = stockPrediction("data/appleTrainData.txt", clusterNum=4)
    apple.reportResult()


    
    # for the case when cluster = 3
    att = stockPrediction("data/attTrainData.txt")
    att.reportResult()
        
    # Case when 2 clusters
    att = stockPrediction("data/attTrainData.txt", clusterNum=2)
    att.reportResult()
       
    # Case when 4 clusters
    att = stockPrediction("data/attTrainData.txt", clusterNum=4)
    att.reportResult()
