import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers 
import statistics as st

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.AUC(name='auc'),
]

'''
Load the training, test and transfer learning dataset of molecular fingerprints along with activity labels.
'''

training_set = pd.read_csv('training_set_FPs.csv', index_col=0)
train_labels = np.asarray(training_set.pop('Activity')).reshape(training_set.shape[0],1)
train_features = np.array(training_set)

ext_test_set =  pd.read_csv('ext_test_set_FPs.csv', index_col=0)
ext_test_set_labels = np.asarray(ext_test_set.pop('Activity')).reshape(ext_test_set.shape[0],1)
ext_test_set_features = np.array(ext_test_set)

tl = pd.read_csv('tl_FPs.csv', index_col=0)
tl_labels = np.asarray(tl.pop('Activity')).reshape(tl.shape[0],1)
tl_features = np.array(tl)

'''
Define the DNN architecture for the regular and transfer learning models
'''

batch_size = 64
n_epochs = 1
patience = 50
drop_rate = 0.25
n_hidden1 =1000
n_hidden2 = 500
learning_rate = 0.001

def dnn_normal():
    model = Sequential()
    model.add(Dense(n_hidden1, activation="relu", input_shape=(1024,)))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_hidden2, activation="relu"))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation="sigmoid"))
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=[METRICS])
    checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model_DNN.keras")
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid, y_valid),
                         callbacks=[checkpoint_cb, early_stopping_cb])
    print()
    if refs == 2:
        model.save("TL_model.keras")

    print("Validation Results")
    validation_results = model.evaluate(X_valid, y_valid,
                                  batch_size=batch_size, verbose=0)

    print("External Test Set Results")
    ext_test_results = model.evaluate(ext_test_set_features, ext_test_set_labels,
                                  batch_size=batch_size, verbose=0)
 
    return validation_results, ext_test_results
        
def dnn_tl2():
    model_m = keras.models.load_model("TL_model.keras")
    model_clone = keras.models.clone_model(model_m)
    model_clone.set_weights(model_m.get_weights())
    model_new = keras.models.Sequential(model_clone.layers)
    for layer in model_new.layers[:-3]:
        layer.trainable = False

    adam = optimizers.Adam(learning_rate=learning_rate)
    model_new.compile(loss = "binary_crossentropy", optimizer = adam, metrics=[METRICS])
    checkpoint_cb =keras.callbacks.ModelCheckpoint("best_model_TL.keras")
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model_new.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_valid, y_valid),
                          callbacks=[checkpoint_cb, early_stopping_cb])
    
    print("Validation Results")
    validation_results = model_new.evaluate(X_valid, y_valid,
                                  batch_size=batch_size, verbose=0)

    print("External Test Set Results")
    ext_test_results = model_new.evaluate(ext_test_set_features, ext_test_set_labels,
                                  batch_size=batch_size, verbose=0)

    return validation_results, ext_test_results

'''
Split the TL data into training and validation sets and run the model
'''
X_train, X_valid, y_train, y_valid = train_test_split(tl_features, tl_labels, test_size=0.10, shuffle=True)
refs = 2
print("TL DNN", X_train.shape, y_train.shape, flush=True)
dnn_normal()

dnn_results = []
tl_results = []

dnn_validation = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])
dnn_ext_results = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])
tl_ext_results = pd.DataFrame(columns=['Loss','TP','FP','TN','FN','AUC'])

'''
Build DNN (1024:1000:500:1) with 90:10 train:test of the smaller dataset, serve as reference for TL and then perform TL
'''
for repeats in range(10):
    X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_labels, test_size=0.10, shuffle=True)

    refs = 0
    print("Small DNN Training Repeats = ",repeats, X_train.shape, y_train.shape, flush=True)    
    dnn_results.append(dnn_normal())
    dnn_validation.loc[repeats,:] = dnn_results[repeats][0]
    dnn_ext_results.loc[repeats,:] = dnn_results[repeats][1]

    print("TL Repeats = ",repeats, X_train.shape, y_train.shape, flush=True)
    tl_results.append(dnn_tl2())
    tl_ext_results.loc[repeats,:] = tl_results[repeats][1]
   
dnn_validation.to_csv('dnn_validation.csv')
dnn_ext_results.to_csv('dnn_ext.csv') 
tl_ext_results.to_csv('tl_ext.csv')




















