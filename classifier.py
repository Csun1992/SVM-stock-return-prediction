import numpy as np
from collections import Counter
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn.decomposition import KernelPCA as pca
from sklearn.metrics import f1_score, precision_score, recall_score

# This is a virtual class of classifier
# Other classifiers will inherit from this class and rewrite the train() method

class Classifier(object):
    def __init__(self, microDataLoc, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc

    def cluster(self, clusterNum):
        data = np.loadtxt(self.macroDataLoc)
        cleanData = preprocessing.scale(data)
        kmeans = cluster.KMeans(n_clusters=clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(kmeans.labels_)
        return groupNum

    def prepareData(self, clusterNum):
        data = np.loadtxt(self.microDataLoc)
        groupNum = self.cluster(clusterNum)
        minSize = min(Counter(groupNum).values()) # find the smallest sample size among all groups
        nComponents = min(max(minSize / 30, 1), np.size(data, 1)) # each feature needs 30 samples
        labels = data[:, -1]
        data = pca(n_components = nComponents, kernel = 'rbf').fit_transform(data[:, :-1])
        group, label = [], []
        for i in range(clusterNum):
            group.append(data[groupNum==i])
            label.append(labels[groupNum==i])
        return (group, label)

    def trainTestSplit(self, clusterNum):
        train, test, trainLabel, testLabel = [], [], [], []
        group, label = self.prepareData(clusterNum)
        for i in range(clusterNum):
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
        clf, test, testLabel, clusterNum = self.train()
        f1 = []
        for i in range(clusterNum):
            pred = clf[i].predict(test[i]) 
            individualF1 = precision_score(testLabel[i], pred)
            f1.append(individualF1)
        return (f1, clusterNum)

    def reportResult(self):
        f1, clusterNum = self.test()
        print "For the case when cluster = " + str(clusterNum) + ' :'
        for i in range(clusterNum):
            print "group NO." + str(i+1) + " f1 score is"
            print  round(f1[i], 2)
        print '\n'
        return f1 
