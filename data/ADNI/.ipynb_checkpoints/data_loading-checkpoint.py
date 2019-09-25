#%% Necessary Packages

import numpy as np

#%%

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#%% Google Data Import
def google_data(seq_length):

    x = np.loadtxt('/home/vdslab/Documents/Jinsung/2019_Research/NIPS/TGAN/Data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
    x = x[::-1]
    x = MinMaxScaler(x)
    
    # Dataset build
    dataX = []
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
        
    dataX = outputX.copy()
    
    return outputX
  
#%%
    
def appliance_data (seq_length):
    
    x = np.loadtxt('/home/vdslab/Documents/Jinsung/2019_Research/NIPS/TGAN/Data/energydata_complete.csv', delimiter = ",",skiprows = 1)
    x = x[::-1]
    x = MinMaxScaler(x)
    
    # Dataset build
    dataX = []
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])    
    
    dataX = outputX.copy()
    
    return outputX
  
