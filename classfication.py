import numpy as np
from sklearn import preprocessing, cluster, model_selection, svm


clusterData = np.loadtxt('data/clusterData.txt')
cleanData = preprocessing.scale(clusterData)

kmeans = cluster.KMeans(n_clusters=3, random_state=11).fit(cleanData)

labels = np.array(kmeans.labels_)

appleData = np.loadtxt('data/appleTrainData.txt') 
aggApple = appleData[:, :-1]
aggResult = appleData[:, -1]


group1 = appleData[labels==0, :-1]
result1 = appleData[labels==0, -1]
group2 = appleData[labels==1, :-1]
result2 = appleData[labels==1, -1]
group3 = appleData[labels==2, :-1]
result3 = appleData[labels==2, -1]
group4 = appleData[labels==3, :-1]
result4 = appleData[labels==3, -1]

fold = 5 
aggClf = svm.SVC()
clf = [svm.SVC() for i in range(4)]
aggScore = model_selection.cross_val_score(aggClf, aggApple, aggResult, cv=fold)
scores1 = model_selection.cross_val_score(clf[0], group1, result1, cv=fold)
scores2 = model_selection.cross_val_score(clf[1], group2, result2, cv=fold)
scores3 = model_selection.cross_val_score(clf[2], group3, result3, cv=fold)
    
print aggScore.mean()    
print aggScore.std()
print '\n'

print scores1.mean()
print scores1.std()
print '\n' 
print scores2.mean()
print scores2.std()
print '\n' 

print scores3.mean()
print scores3.std()
print '\n' 


print '\n'    
print np.size(result1)
print np.size(result2)
print np.size(result3)



