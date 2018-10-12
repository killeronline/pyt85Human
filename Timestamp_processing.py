import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 

worksheet = pd.read_excel('data/AB.xlsx')
raw_data = np.array(worksheet)
raw_data = raw_data[:,2:] # Removing Symbol, Series
offset = 7 # Max Past Days Used In Formulae

feature_count = 13
calc_data = np.zeros((len(raw_data)-offset,feature_count),dtype=object)
for i in range(offset,len(raw_data)):
    ind = i - offset
    calc_data[ind][0] = raw_data[i][0].dayofweek
    calc_data[ind][1] = raw_data[i][1]
    calc_data[ind][2] = raw_data[i][2]
    calc_data[ind][3] = raw_data[i][3]
    calc_data[ind][4] = raw_data[i][4]
    calc_data[ind][5] = raw_data[i][5]
    calc_data[ind][6] = raw_data[i][6]
    calc_data[ind][7] = raw_data[i][7]
    calc_data[ind][8] = raw_data[i][8]
    calc_data[ind][9] = raw_data[i][9]
    calc_data[ind][10] = raw_data[i][10]
    calc_data[ind][11] = raw_data[i][11]
    calc_data[ind][12] = raw_data[i][12]


    
    
    
    

'''
day of the week t.dayofweek...Monday 0 .....Sunday 6.....primary
day of the month t.day....secondary
week of the month (t.week%4)+1...secondary
'''
    