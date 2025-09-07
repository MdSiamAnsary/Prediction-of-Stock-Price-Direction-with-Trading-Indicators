# Importing necessary packages
import talib
import yfinance as yf
import numpy as np
import pandas as pd
from collections import Counter
import imblearn.over_sampling
from imblearn.over_sampling import RandomOverSampler


# If the CSV file already exists, delete it
import os
if os.path.exists("file.csv"):
  os.remove("file.csv")


# Read the data
data = pd.read_csv("2021.csv")
print(data.shape)

#Moving Average
ma = talib.MA(data['Close'], timeperiod=30, matype=0)
data['MA'] = ma
data['MA'] = data['MA'].fillna(0)

# Balance of Power
bop = talib.BOP(data['Open'], data['High'], data['Low'], data['Close'])
data['BOP'] = bop
data['BOP'] = data['BOP'].fillna(0)

# Money Flow Index
mfi = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
data['MFI'] = mfi
data['MFI'] = data['MFI'].fillna(0)

# Momentum
mom = talib.MOM(data['Close'], timeperiod= 10)
data['MOM'] = mom
data['MOM'] = data['MOM'].fillna(0)

# Rate of Change
roc = talib.ROC(data['Close'], timeperiod=10)
data['ROC'] = roc
data['ROC'] = data['ROC'].fillna(0)

#Relative Strength Index
rsi = talib.RSI(data['Close'], timeperiod=14)
data['RSI'] = rsi
data['RSI'] = data['RSI'].fillna(0)

# Weighted Close Price
wcp = talib.WCLPRICE(data['High'], data['Low'], data['Close'])
data['WCP'] = wcp
data['WCP'] = data['WCP'].fillna(0)

# Average Price
ap = talib.AVGPRICE(data['Open'], data['High'], data['Low'], data['Close'])
data['AP'] = ap
data['AP'] = data['AP'].fillna(0)

# Median Price
mp = talib.MEDPRICE(data['High'], data['Low'])
data['MP'] = mp
data['MP'] = data['MP'].fillna(0)

# Typical Price
tp = talib.TYPPRICE(data['High'], data['Low'], data['Close'])
data['TP'] = tp
data['TP'] = data['TP'].fillna(0)

# On Balance Volume
obv = talib.OBV(data['Close'], data['Volume'])
data['OBV'] = obv
data['OBV'] = data['OBV'].fillna(0)

# Dropping Noisy Columns
data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)

# Class of each stock
data['Class'] = 0
# If value = 0 , market stable
# If value = 1, overbought, market upwards (bullish)
# If value = -1, oversold, market downwards (bearish)

# Column names of the dataframe
print(data.columns)

# DataFrame to Numpy Array 
arr = data.to_numpy()

# Shape of the array
print(arr.shape)



# If MFI > 80 , overbought
# If MFI < 20 , oversold
# If RSI > 70 , overbought
# If RSI < 30 , oversold

for x in arr:
    if(x[7] > 80 or x[10] > 70):
        x[16] = 1
    elif(x[7] < 20 or x[10] < 30):
        x[16] = -1

# Numpy array to dataframe
data = pd.DataFrame(arr)

# Check number of instances of each class
print(data[16].value_counts())

# Making the dataset balances

# Undersampling starts 
pos_class_len = len(data[data[16]==1])
pos_class_indices= data[data[16]==1].index

zero_class_len = len(data[data[16]==0])
zero_class_indices= data[data[16]==0].index
random_zero_class_indices = np.random.choice(zero_class_indices, pos_class_len, replace=False)

neg_class_indices= data[data[16]==-1].index

sample_indices = np.concatenate([random_zero_class_indices, pos_class_indices, neg_class_indices])
sample = data.loc[sample_indices]
# Undersampling done

print(sample[16].value_counts())

# Oversampling starts 
ros = RandomOverSampler()

X = sample[[0,1,2,3,
            4,5,6,7,
            8,9,10,11,
            12,13,14,15]]
Y = sample[[16]]

x_ros, y_ros = ros.fit_resample(X,Y.values.ravel())


features_data = pd.DataFrame(x_ros)
target_data = pd.DataFrame(y_ros)
print(target_data[0].value_counts())
result = pd.concat([features_data, target_data], axis=1)
# Oversampling done


print(result.shape)

# Saving the dataframe in a csv file
result.to_csv("file.csv")












