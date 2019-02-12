import numpy as np
from sys import exit
from collections import Counter
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn.decomposition import KernelPCA as pca
from sklearn.metrics import f1_score, precision_score, recall_score

# This is a virtual class of classifier
# Other classifiers will inherit from this class and rewrite the train() method

class Classifier(object):
    def __init__(self, microDataLoc, clusterNum = 3, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc
        self.clusterNum = clusterNum

    # load and cluster the dataset into number of groups equal clusterNum 
    def cluster(self): 
        data = np.loadtxt(self.macroDataLoc) # data for clustering with k-mean
        cleanData = preprocessing.scale(data)
        kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(kmeans.labels_)
        return groupNum

    def prepareData(self):
        data = np.loadtxt(self.microDataLoc) # data for classification with specified classifier
        groupNum = self.cluster()
        minSize = min(Counter(groupNum).values()) # find the smallest sample size among all groups
#nComponents = 1 #min(max(minSize / 30, 1), np.size(data, 1)) # each feature needs 30 samples
        labels = data[:, -1]
        data = data[:,:-1]
#data = pca(n_components = nComponents, kernel = 'rbf').fit_transform(data[:, :-1])
        group, label = [], []
        for i in range(self.clusterNum):
            group.append(data[groupNum==i])
            label.append(labels[groupNum==i])
        return (group, label)

    def trainTestSplit(self):
        train, test, trainLabel, testLabel = [], [], [], []
        group, label = self.prepareData()
        for i in range(self.clusterNum):
            trainData, testData, trainLabelData, testLabelData = model_selection.train_test_split(group[i],
                    label[i], test_size=0.1, random_state=11)
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
        f1 = []
        for i in range(self.clusterNum):
#            print 'for the ' + str(i) + 'th cluster'
#            print sum(testLabel[i]) / len(testLabel[i])
#            print "\n" 
            pred = clf[i].predict(test[i]) 
#            print testLabel[i]
#            print pred
            individualF1 = f1_score(testLabel[i], pred)
            individualPrec = precision_score(testLabel[i], pred)
            individualRecall = recall_score(testLabel[i], pred)
            f1.append((round(individualF1, 2), round(individualPrec, 2), round(individualRecall, 2)))
#            print f1[i]
        return f1

    def reportResult(self):
        f1 = self.test()
        print "For the case when cluster = " + str(self.clusterNum) + ' :'
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " f1 score is"
            print f1[i]
        print '\n'
        return f1 
