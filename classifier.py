import numpy as np
from collections import Counter
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn.decomposition import KernelPCA as pca
from sklearn.metrics import f1_score 

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
        minSize = min(Counter(groupNum).values()) # find the smallest sample size among all groups
        nComponents = min(max(minSize / 30, 1), np.size(data, 1)) # each feature needs 30 samples
        labels = data[:, -1]
        data = pca(n_components = nComponents, kernel = 'linear').fit_transform(data[:, :-1])
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
                    label[i], test_size=0.3, random_state=11)
            train.append(trainData)
            test.append(testData)
            trainLabel.append(trainLabelData)
            testLabel.append(testLabelData)
#print testLabel    
        return (train, test, trainLabel, testLabel)
            
    def train(self):
        pass        # virtual method to be overwritten in each specific classifier
                    # must return a tuple of classifier instances, test data and test labels

    def test(self):
        clf, test, testLabel, cv = self.train()
        f1 = []
        for i in range(self.clusterNum):
            pred = (clf[i].predict(test[i]) == 1)
            f1.append(f1_score(testLabel[i], pred, average = 'micro'))
#caseError = sum([i != j for (i,j) in zip(testLabel[i], pred)])
#error.append(float(caseError)/len(pred))
        return (f1, cv)

    def reportResult(self):
        f1, cv = self.test()
        print "the cross validation error is " 
        print cv
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " F1 score is"
            print f1[i]
        print '\n'
        return f1 
