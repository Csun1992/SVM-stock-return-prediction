import numpy as np
from sklearn import preprocessing, cluster, model_selection, svm

# This is a virtual class of classifier
# Other classifiers will inherit from this class and rewrite the train() method

class Classifier(object):
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
        pass        # virtual method to be overwritten in each specific classifier
                    # must return a tuple of classifier instances, test data and test labels

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
