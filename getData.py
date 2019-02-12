from sklearn import preprocessing, cluster, model_selection, svm
import numpy as np
import sys

def getStockPrice(fileName):
    price = []
    size = []
    with open(fileName) as f:
        for line in f:
            item = line.rstrip().split(',')
            price.append(float(item[-2]))
            size.append(float(item[-1]) / 1000000)
    return price, size[2:-1]

def getInputData(fileName):        
    fileName = 'data/' + fileName
    price, size = getStockPrice(fileName)
    threeMonthMA = [(i+j+k)/3 for i,j,k in zip(price, price[1: ], price[2: ])]
    del threeMonthMA[-1]
    twoMonthMA = [(i+j)/2 for i,j in zip(price[1: ], price[2: ])]
    del twoMonthMA[-1]
    stockReturn = [(j-i)/i for i,j in zip(price[2: ], price[3: ])]
#    del stockReturn[-1]
    classification = np.array(map(int, [i > 0.005 for i in stockReturn])).reshape(-1, 1)
    inputData = np.array([price[2:-1], twoMonthMA, threeMonthMA]).T
#   inputData = preprocessing.scale(inputData)
    inputData = np.concatenate((inputData, classification), axis = 1)
#    inputData = np.array([price[2:-1], twoMonthMA, threeMonthMA, classification]).T
    fileName = fileName + 'TrainData.txt'
    np.savetxt(fileName, inputData)
    return inputData


if __name__ == '__main__':
    capacity = []
    with open("data/capacity", 'r') as f:
        for line in f:
            capacity.append(line.rstrip().rstrip('\n').split(',')[1])
    capacity = [float(i) for i in capacity]

    unemployment = []
    with open("data/unemployment", 'r') as f:
       for line in f:
           unemployment.append(map(float, line.rstrip().rstrip('\n').split(' ')))
    unemployment = [item for sublist in unemployment for item in sublist]

    cpi = []
    with open("data/cpi", 'r') as f:
        for line in f:
            cpi.append(line.rstrip().rstrip('\n').split(',')[1])
    cpi = map(float, cpi)
    inflation = [(j - i) / i for i, j in zip(cpi[:-1], cpi[1:])]
    
    djia = []
    with open("data/djia", 'r') as f:
        for line in f:
            djia.append(float(line.rstrip()))
    djia = [(j-i)/i for i,j in zip(djia, djia[1:])]
    
    sp = []
    with open('data/sp', 'r') as f:
        for line in f:
            item = line.rstrip().split(',')
            sp.append(float(item[-2]))
    sp = [(j-i)/i for i,j in zip(sp, sp[1:])]
    
    clusterData = np.array([unemployment, inflation, capacity]).T
    clusterData = np.array([unemployment, inflation, capacity]).T
    np.savetxt('data/clusterData.txt', clusterData)
    
    fileName = ['microsoft', 'apple', 'att', 'ford', 'sony', 'gap', 'fedex', 'mcdonalds', 'nike',
    'tiffany', 'homeDepot', 'walmart', 'cocaCola', 'avon', 'oracle', 'ibm', 'intel',
    'harley-davidson', 'toyota', 'honda', 'boeing', 'jpmorgan', 'boa', 'amgen', 'hermanMiller',
    'nissan', 'generalElectric', 'nextEra', 'conocoPhillips', 'bakerHughes', 'dukeEnergy', 'chevron']
    for i in fileName:
        getInputData(i)
