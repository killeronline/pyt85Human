import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 

fc = 0 # count
def fi():
    global fc
    fc += 1
    return fc - 1

worksheet = pd.read_excel('data/AB.xlsx')
raw_data = np.array(worksheet)
raw_data = raw_data[:,2:] # Removing Symbol, Series
offset = 7 # Max Past Days Used In Formulae

feature_count = 14
calc_data = np.ones((len(raw_data)-offset,feature_count),dtype=object)
calc_labs = np.ones((len(raw_data)-offset,1),dtype=object)
for i in range(offset,len(raw_data)):
    ind = i - offset    
    fc=0
    # Labels
    calc_labs[ind] = raw_data[i][6]                 # Close Price
    
    # Features
    #calc_data[ind][fi()] = raw_data[i][0].dayofweek+1  # Monday=1, Tuseday=2 ...
    calc_data[ind][fi()] = raw_data[i][1]              # Prev Close
    calc_data[ind][fi()] = raw_data[i][2]              # Open Price    
    # Key Feature Count = 3
    
    t = 1 
    for corr in range(1,4):
        t = corr # correlation depth
        # First Correlation                                # T - 1            
        #calc_data[ind][fi()] = raw_data[i-t][1]              # Prev Close
        calc_data[ind][fi()] = raw_data[i-t][2]              # Open Price
        calc_data[ind][fi()] = raw_data[i-t][3]              # High Price 
        calc_data[ind][fi()] = raw_data[i-t][4]              # Low Price
        #calc_data[ind][fi()] = raw_data[i-t][5]              # Last Price
        calc_data[ind][fi()] = raw_data[i-t][6]              # Close Price 
        #calc_data[ind][fi()] = raw_data[i-t][7]              # Avg Price
        #calc_data[ind][fi()] = raw_data[i-t][8]              # TTQ
        #calc_data[ind][fi()] = raw_data[i-t][10]             # NOT
        #calc_data[ind][fi()] = raw_data[i-t][11]             # DQ
        #calc_data[ind][fi()] = raw_data[i-t][12]             # DQ TO TQ %        
        # Corr Feature Count = 11

# Seperating data into training data and testing data
train_percentage = 50
train_end_index = int(len(calc_data)*train_percentage/100)
train_data =    np.array(calc_data[:train_end_index,:],dtype=float)
train_labels =  np.array(calc_labs[:train_end_index,:],dtype=float)
test_data =     np.array(calc_data[train_end_index:,:],dtype=float)
test_labels =   np.array(calc_labs[train_end_index:,:],dtype=float)


# Feature Normalization
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# Model Building
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),    
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 2000

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])


import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [xxx$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)


[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1)) #1000))


test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [xxx$]')
plt.ylabel('Predictions [xxx$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [xxx$]")
_ = plt.ylabel("Count")


print(test_predictions)    
print(test_labels.flatten())    
    
    
    
    
    
        
    

