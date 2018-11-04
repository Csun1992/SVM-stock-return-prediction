import numpy as np

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


def getStockPrice(fileName):
    price = []
    size = []
    with open(fileName) as f:
        for line in f:
            item = line.rstrip().split(',')
            price.append(float(item[-2]))
            size.append(float(item[-1]))
    return price, size[2:]

applePrice, appleSize = getStockPrice("data/apple")
appleThreeMonthMA = [(i+j+k)/3 for i,j,k in zip(applePrice, applePrice[1:], applePrice[2:])]
appleTwoMonthMA = [(i+j)/2 for i,j in zip(applePrice[1:], applePrice[2:])]
appleReturn = [(j-i)/i for i,j in zip(applePrice[1:], applePrice[2:])]
classification = map(int, [i>0 for i in appleReturn])
appleData = np.array([applePrice[2:], appleTwoMonthMA, appleThreeMonthMA, appleSize, classification]).T
np.savetxt('data/appleTrainData.txt', appleData)
