import pandas as pd
import numpy as np

# label_idx: ADNI11 or ADNI13
# train_rate = 0.8
# Window = 5    

def ADNI_Process (label_idx, train_rate, Window):

    #%% Data Preprocess
    # Read Original Data
    oriData = pd.read_csv('/home/vdslab/Documents/Jinsung/2019_Research/ICML/INVASET/Real/ADNI/Data_Preprocess/ADNI_Data.csv')
    
    # Extract ID and Unique ID (Uni_ID)
    ID = np.asarray(oriData['ID'])
    Uni_ID = np.intersect1d(ID,ID)
    
    # Parameters
    Total_No = len(ID)
    No = len(Uni_ID)
    
    # Find the sequence length for each data
    Seq_Len = np.zeros([No,])
    
    for i in range(No):
        Seq_Len[i] = len(np.where(ID == Uni_ID[i])[0])
    
    # Find IDs with less than window length of sequences
    Remove_ID = Uni_ID[np.where(Seq_Len<Window)]
    
    # Remove IDs with less than window length of sequences
    A = list()
    for i in range(Total_No):
        if (ID[i] in Remove_ID):
            A.append(i)
    
    newData = oriData.drop(A)
    
    #%% Same thing with newData (with no sequence with less than window length)
    
    # ID
    ID = np.asarray(newData['ID'])
    Uni_ID = np.intersect1d(ID,ID)
    
    # Parameters
    Total_No = len(ID)    
    No = len(Uni_ID)
    
    #%% Feature / Label
    Label = newData[['ID','ADAS11','ADAS13','Event_1']]
    Feature = newData.drop(columns=['ID','ADAS11','ADAS13','Event_1','Event_2','Event_3','Event_4','Event_5'])
    
    Label = np.asarray(Label)
    Feature = np.asarray(Feature)
    
    # Feature normalization
    for k in range(len(Feature[0,:])):
        Feature[:,k] = Feature[:,k] - np.min(Feature[:,k])
        Feature[:,k] = Feature[:,k] / (np.max(Feature[:,k]) + 1e-8)
    
    # Final Feature and Label extraction with same window size    
    Final_Feature = list()
    Final_Label = list()
    
    # For each patient
    for i in range(No):
      
        # Find the entire sequence of that ID
        idx = np.where(Label[:,0] == Uni_ID[i])[0]
        
        Temp_Feature = Feature[idx,:]
        Temp_Label = Label[idx,:]
        
        # Cut them in the window size
        for j in range(len(idx)-Window+1):
          
            # Feature extraction
            Final_Feature.append(Temp_Feature[j:j+Window,:])
            
            # Label extraction
            if (label_idx == 'ADAS11'):
                Final_Label.append(Temp_Label[j:j+Window,1])
            elif (label_idx == 'ADAS13'):
                Final_Label.append(Temp_Label[j:j+Window,2])
            elif (label_idx == 'Event'):
                Final_Label.append(Temp_Label[j:j+Window,3])
                
    #%% Train / Test Division
    No = len(Final_Label)
    Train_No = int(No * train_rate)
    idx = np.random.permutation(No)
    
    train_idx = idx[:Train_No]
    test_idx = idx[Train_No:No]
    
    trainX = list(Final_Feature[i] for i in train_idx)
    trainY = list(Final_Label[i] for i in train_idx)
    
    testX = list(Final_Feature[i] for i in test_idx)
    testY = list(Final_Label[i] for i in test_idx)    
    
    return trainX, trainY, testX, testY