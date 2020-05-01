import pandas as pd 
import quandl
from pprint import pprint
import math

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
## Using 10% of the data for prediction (using data from 10 days ago for prediction)
forecast_out = int(math.ceil(0.01*len(df)))

df ['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
#pprint(df.tail())

pprint(df.head())


 
