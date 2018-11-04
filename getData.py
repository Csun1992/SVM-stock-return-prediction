import numpy as np

unemployment = []
with open("unemployment", 'r') as f:
   for line in f:
       unemployment.append(map(float, line.rstrip().rstrip('\n').split(' ')))
unemployment = [item for sublist in unemployment for item in sublist]

inflation = []
with open("inflation", 'r') as f:
    for line in f:
        inflation.append(line.rstrip().rstrip('\n').split('   '))
inflation = map(float, [item.rstrip('%') for sublist in inflation for item in sublist])

djia = []
with open("djia", 'r') as f:
    for line in f:
        djia.append(float(line.rstrip()))
djia = [(i-j)/j for i,j in zip(djia, djia[1:])]

sp = []
with open('sp', 'r') as f:
    for line in f:
        item = line.rstrip().split(',')
        sp.append(float(item[-2]))
sp = [(i-j)/j for i,j in zip(sp, sp[1:])]

clusterData = np.array([unemployment, inflation, djia, sp]).T
np.savetxt('data/clusterData.txt', clusterData)
