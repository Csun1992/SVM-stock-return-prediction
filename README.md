# Project Purpose
This project constitutes the first step in building a full-fledged recommendation system which manages the allocation of assets of customers based on their risk tolerance, and the profitability of the investments etc.

In this repository we predict stock price moving direction with K-means clustering and support vector machine

# Dependencies
- Numpy
- sckit-learn

# Detailed Description of the Machine Learning Algorithm
In this project, we predict the direction of stock price movement in the future months. So essentially, it is a classification problem.

First, we cluster time periods into three clusters. The reason is because business cycles consist of 4 phases: 
recovery, prosperity, recession and depression. The reason we use three clusters instead of four is due to the 
limitation of the sample size.

The factors we use to cluster the time periods are macroeconomic factors: inflation rate, unemployment rate,
rate of change of Dow Jones Industrial Average and rate of change of S & P 500. 

Then for each cluster, we built a support vector machine with idiosyncratic factors for each stock: three-month 
moving average of stock price, two month moving average of stock price, stock price in current month and current
fundamental value of the company. We try to catch the trend of the stock price movement with these factors.

The machine learning technique we chose here will take the macroeconomic environment into consideration when predicting the stock price movement. It is helpful in improving prediction precision since stock prices may behave differently during different periods of business cycles.

The data we used are all monthly data from Jan. 01 1990 to Sep. 01 2018. We avoided the noisy daily or weekly data.  

The stocks we used as examples include Apple,ATT.

# Components of the Repository
1. The folder named *data* which contains the data for clustering and svm classification. The raw clustering data are mainly from different government websites. The raw data for classification are from yahoo finance.
2. The python file named *getData.py*. This file contains functions that transforms the raw data into format suitable for training. Since the raw data are taken from different sources, we wrote the ad hoc functions to transform the data. And then save the transformed data into *data* folder as well.
3. The python file *svm.py*. This is the main file. It implements the algorithm. In this file, we created an *svmStockPred* that inherits from *Classifier* -- a virtual class that all classifiers inherit form which can be constructed with the information about the location of the transformed clustering and classification data files. If want to find the final result about the testing error, we can simply call *reportResult()* method embedded in the object. It automatically runs the algorithm and reports tesing error. By default we set have "80-20" split for the training and tesing data.<br />
In addition, for comparison, we implemented a *svmNoClustering* object that inherits from the *svmStockPred*. In this derived class, we simply implement a svm with the clustering and classification data combined. Similar to the parent class, we can simply call the *reportResult()* method in this derived class to find out the testing errors.

# Difficulty
The best features for predicting the stock movement. 

# Future Work
1. To compare the statistical difference of the performance between the svm with clustering and the svm without clustering, we are planning to perform certain hypothesis testing. 
2. The data amount is still relatively small if we cluster them into three or four clusters. Thus we will perform experiment on the weekly data.
3. Combine the two svm's into one with ensemble methods to see how big improvement we can have.
