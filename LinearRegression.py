import pandas as pd 
import quandl, math
from pprint import pprint
import numpy as np
from sklearn import preprocessing,  svm 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
## get stock data
df = quandl.get('WIKI/GOOGL')
## Select only attribute needed
## Linear regression does not seek out patterns between features
## Adj. Open is a feature = attribute 
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

## Reduce the number of features by defining the relahinship between them 
## High low percent change
df['HL_PCT'] = 100 * (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"]
## Daily percent change 
df['PCT_Change'] = 100 * (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"]
## volume is how many trades occured that day 
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

pprint(df.head())

forecast_col = "Adj. Close"
## instead of deleting NA data
df.fillna(-99999, inplace = True)
## Using 1% of the data for prediction (using data from 10 days ago for prediction)
forecast_out = math.ceil(0.01*len(df))
df ['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
#pprint(df.tail())
#pprint(df.head())

X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])

X = preprocessing.scale(X)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)

print (accuracy)




 
