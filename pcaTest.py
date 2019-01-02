from getData import getInputData
import numpy as np
from sklearn.decomposition import KernelPCA 

dat = getInputData('apple')

pca = KernelPCA(n_components = 4, kernel = 'linear')
XTransformed = pca.fit_transform(dat)
print XTransformed 

