import pandas as pd 
from datetime import datetime
import quandl, math, datetime
from pprint import pprint
import numpy as np
from sklearn import preprocessing,  svm 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
from dateutil.parser import parse
style.use('ggplot')


## get stock data


#df = quandl.get('WIKI/GOOGL')
#df.to_csv('WIKI_GOOGL.csv')
#with open('WIKI_GOOGL.pickle','wb') as f: 
#	pickle.dump(df,f)
df = pd.read_csv('WIKI_GOOGL.csv')
#df.set_index('Date')
#pickle_in = open('WIKI_GOOGL.pickle','rb')
#df = pickle.load(pickle_in)
## Select only attribute needed
## Linear regression does not seek out patterns between features
## Adj. Open is a feature = attribute 
df = df.set_index('Date')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

## Reduce the number of features by defining the relahinship between them 
## High low percent change
df['HL_PCT'] = 100 * (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"]
## Daily percent change 
df['PCT_Change'] = 100 * (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"]
## volume is how many trades occured that day 
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]


forecast_col = "Adj. Close"
## instead of deleting NA data
df.fillna(-99999, inplace = True)
print(df.tail())
## Using 1% of the data for prediction (using data from 10 days ago for prediction)
forecast_out = math.ceil(0.01*len(df))
df ['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[ -forecast_out:]
X = X[:-forecast_out] 

df.dropna(inplace=True)
print(df.tail())
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
#print (accuracy)
# prediction for the next 35 days 

forecast_set = clf.predict(X_lately)
print (forecast_set, accuracy, forecast_out)

#plotting our prediction
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
last_unix = last_date.timestamp() 
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: 
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
	
# plotting
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()















































 
