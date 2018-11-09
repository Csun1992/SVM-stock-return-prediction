import numpy as np
import sys
def getStockPrice(fileName):
    price = []
    size = []
    with open(fileName) as f:
        if fileName == 'data/apple':
            for line in f:
                item = line.rstrip().split(',')
                price.append(float(item[-2]))
                size.append(float(item[-1]))
        else:
            for line in f:
                item = line.rstrip().split('\t')
                price.append(float(item[-2]))
                size.append(float(item[-1]))
    return price, size[2:-1]

def getInputData(fileName):        
    fileName = 'data/' + fileName
    price, size = getStockPrice(fileName)
    threeMonthMA = [(i+j+k)/3 for i,j,k in zip(price, price[1: ], price[2: ])]
    del threeMonthMA[-1]
    twoMonthMA = [(i+j)/2 for i,j in zip(price[1: ], price[2: ])]
    del twoMonthMA[-1]
    stockReturn = [(j-i)/i for i,j in zip(price[2: ], price[3: ])]
    classification = map(int, [i>0 for i in stockReturn])
    inputData = np.array([price[2:-1], twoMonthMA, threeMonthMA, size, classification]).T
    fileName = fileName + 'TrainData.txt'
    np.savetxt(fileName, inputData)
    return inputData


if __name__ == '__main__':
    unemployment = []
    with open("data/unemployment", 'r') as f:
       for line in f:
           unemployment.append(map(float, line.rstrip().rstrip('\n').split(' ')))
    unemployment = [item for sublist in unemployment for item in sublist]
    
    inflation = []
    with open("data/inflation", 'r') as f:
        for line in f:
            inflation.append(line.rstrip().rstrip('\n').split('   '))
    inflation = map(float, [item.rstrip('%') for sublist in inflation for item in sublist])
    
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
    
    clusterData = np.array([unemployment, inflation, djia, sp]).T
    np.savetxt('data/clusterData.txt', clusterData)
    
    fileName = 'apple' 
    getInputData(fileName)
    getInputData('att') 
