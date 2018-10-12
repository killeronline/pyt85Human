# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:08:07 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 

worksheet = pd.read_excel('data/AB.xlsx')
data = np.array(worksheet)
data = data[:,2:] # Removing Symbol, Series

# Column Indicies
OPEN  = 2
HIGH  = 3
LOW   = 4
CLOSE = 6
LAST  = 5
AVG   = 7
TTQ   = 8
TURN  = 9
NT    = 10
DQ    = 11
PDQ   = 12

#
UP = 1
DOWN = -1
SIDE = 0

i = 0
tds = 0
hold = 0
acc = 10000
curr = acc
offset = 30 # Max Past Days Used In Formulae

def printb():
    global data
    global i
    print(i," Buying At:",REF(LAST))
    
def prints():
    global data
    global i
    print(i," Selling At:",REF(LAST)) 
    
def REF(INDEX,T=0):
    global data
    global i
    return data[i-T][INDEX]

def TREND(INDEX,PERIODS):
    global data
    global i
    global offset
    if PERIODS > offset :
        PERIODS = offset
    trend = 0
    for j in range(PERIODS,0,-1):        
        if REF(INDEX,j) < REF(INDEX,j-1):
            trend += 1
        elif REF(INDEX,j) > REF(INDEX,j-1):
            trend -= 1    
    if trend == PERIODS :
        return 1 # Up
    elif trend == (-PERIODS) :
        return -1 # Down
    else:
        return 0 # Side
    
    
def b(code):
    global data
    global i
    if code == 1 : # Bullish Meeting Line
        if REF(CLOSE,1)>REF(OPEN,1) and \
 	 	 	REF(OPEN,2)>REF(CLOSE,2) and \
 	 	 	REF(CLOSE,1)==REF(CLOSE,2):
            return True
    if code == 2 : # Bullish Homing Pigeon
        if TREND(CLOSE,5)==DOWN and \
			REF(CLOSE,1)>REF(OPEN,1) and \
 	 	 	REF(CLOSE,2)<REF(OPEN,2) and \
 	 	 	REF(HIGH,1)<REF(OPEN,2) and \
			REF(LOW,1)>REF(CLOSE,2):
           return True
                    
    return False

def s(code):
    global data
    global i
    if code == 1 :
        if REF(CLOSE,1)<REF(OPEN,1) and REF(CLOSE,2)<REF(OPEN,2):
            return True
    if code == 2 :
        if TREND(CLOSE,5) == UP and \
            REF(OPEN,1)>REF(CLOSE,1) and \
            REF(OPEN,2)>REF(CLOSE,2):
            return True
                    
    return False

   

for i in range(offset,len(data)):
    c = 2
    # try to buy
    if acc > 0 and b(c) :
        printb()
        hold = acc/REF(LAST)
        acc = 0
    elif acc == 0 and s(c) :
        prints()
        acc = hold * REF(LAST)        
        curr = acc # current last
        hold = 0    

print('Current ',curr)
print('Holding ',hold)
        
        
  
    
    
















































    
    
    
    
    
        
    

