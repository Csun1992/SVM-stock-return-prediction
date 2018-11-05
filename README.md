# SVM-stock-return-prediction
We predict stock price with K-means clustering and support vector machine

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

The stocks we used as examples include Apple,
