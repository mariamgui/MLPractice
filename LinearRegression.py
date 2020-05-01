import pandas as pd 
import quandl
from pprint import pprint
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



 
