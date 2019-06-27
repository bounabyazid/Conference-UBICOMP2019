#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:54:57 2019

@author: Yazid
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import entropy, chisquare
from scipy.stats.stats import pearsonr   

def Preprocess_Data():
    #https://www.datacamp.com/community/tutorials/joining-dataframes-pandas
    df = pd.read_csv('tweets_original.tsv',delimiter='\t',encoding='utf-8')
    df2 = pd.read_csv('tweets_twins.tsv',delimiter='\t',encoding='utf-8')
    Sentiment = {'Compound1': df['Compound'].tolist(), 'Negative1': df['Negative'].tolist(),'Neutral1': df['Neutral'].tolist(),'Positive1': df['Positive'].tolist(), 
             'Compound2': df2['Compound'].astype(float).tolist(), 'Negative2': df2['Negative'].tolist(),'Neutral2': df2['Neutral'].tolist(),'Positive2': df2['Positive'].tolist(),
             'Locations': df['Locations'].tolist(), 'Timestamp': df['Timestamp'].tolist(), 'Uuid': df['Uuid'].tolist()}

    df3 = pd.DataFrame(Sentiment)
    df3 = df3.replace([np.inf, -np.inf], np.nan)

    df3 = df3.replace('', np.nan)
    df3 = df3.dropna()#removing NaN
   
    #df = df.dropna(subset=['col1', 'col2'], how='all')
    #df3.to_csv('Original_Twin.tsv',sep='\t')
    
    #df4 = pd.merge(df, df2, on = 'Uuid')
    df4 = pd.merge(df, df2, on='Uuid', how='outer')
    return df,df2,df3,df4

def Normalisation(x):#just to remove negative values
    x = x.reshape((len(x), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(x)
    return scaler.transform(x)

def cross_entropy(df,C):
    x = Normalisation(np.asarray(df[C+'1'].tolist()))
    y = Normalisation(np.asarray(df[C+'2'].tolist()))
    return  entropy(x) + entropy(x, y)

def Correlation(df,C):
    x = Normalisation(np.asarray(df[C+'1'].tolist()))
    y = Normalisation(np.asarray(df[C+'2'].tolist()))
    return  pearsonr(x,y)

def Chi_Square(df,C):
    x = Normalisation(np.asarray(df[C+'1'].tolist()))
    y = Normalisation(np.asarray(df[C+'2'].tolist()))
    return chisquare(f_obs=y,f_exp=x)

def MAE(df,C):
    x = np.asarray(df[C+'1'].tolist())
    y = np.asarray(df[C+'2'].tolist())
    return  mean_absolute_error(x,y)
    
def Loss_Measures():
    df,df2,df3 = Preprocess_Data()
    df = df3
    #CrossEntropy
    print('________________Cross Entropy________________')
    
    print ('Compound',cross_entropy(df,'Compound'))
    print ('Negative',cross_entropy(df,'Negative'))
    print ('Positive',cross_entropy(df,'Positive'))
    print ('Neutral ',cross_entropy(df,'Neutral'))
    
    #Correlation
    print('_________________Correlation_________________')

    print ('Compound',Correlation(df,'Compound'))
    print ('Negative',Correlation(df,'Negative'))
    print ('Positive',Correlation(df,'Positive'))
    print ('Neutral ',Correlation(df,'Neutral'))
    
    #Chi Square
    print('_________________Chi-Square__________________')

    print ('Compound',Chi_Square(df,'Compound'))
    print ('Negative',Chi_Square(df,'Negative'))
    print ('Positive',Chi_Square(df,'Positive'))
    print ('Neutral ',Chi_Square(df,'Neutral'))
    
    #Correlation
    print('_____________________MAE_____________________')

    print ('Compound',MAE(df,'Compound'))
    print ('Negative',MAE(df,'Negative'))
    print ('Positive',MAE(df,'Positive'))
    print ('Neutral ',MAE(df,'Neutral'))

    return df
df,df2,df3,df4 = Preprocess_Data()
#df = Loss_Measures()
