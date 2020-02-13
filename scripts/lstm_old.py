from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import LSTM
import numpy as np
from predictive_imputer import predictive_imputer
import impyute as impy
import sys

AADT_test = []
AADT_pred = []

ATRs = np.array([1])

loss = []
val_loss = []
hrmse = []
hmape = []
armse = []
amape = []
missing_value = []
values2 = []

dataset = read_csv('1.csv', usecols=range(0,2))
values = dataset.values
values = values.astype('float')

values2 = np.append(values2,values[0:78768,0])
values2 = np.append(values2,values[69864:78768,1])
values2 = values2.reshape(87672,1)

missing_value = np.append(missing_value, (len(values2)-np.count_nonzero(values2))*100/len(values2))

values[values == 0] = np.nan
#imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
#values = imputer.fit(values).transform(values.copy())
values = impy.mean(values)
values[values < 0] = 0

n_train = 61224
n_test = 17544

values = values.reshape(values.shape[0],2)

n_col = values.shape[1]

train = values[:n_train, :]
test = values[n_train:, :]

#train = np.log(1+train)
#test = np.log(1+test)

temp1 = np.amax(train) - np.amin(train)
temp2 = np.amin(train)
scaler1 = np.zeros((1,n_col))
scaler2 = np.zeros((1,n_col))
scaler1[0,0] = temp1
scaler1[0,1] = temp1
scaler2[0,0] = temp2
scaler2[0,1] = temp2
train = (train - scaler2) / scaler1
test = (test - scaler2) / scaler1
train = train.astype('float')
test = test.astype('float')
print(scaler1, scaler2)

# split into input and outputs
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, 1))
test_X = test_X.reshape((test_X.shape[0], 1, 1))
train_y = train_y.reshape((train_y.shape[0], 1))
test_y = test_y.reshape((test_y.shape[0], 1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(1, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dropout(0.2))
#model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='Adam')
# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=672, validation_data=(test_X, test_y), verbose=2, shuffle=False)
loss = np.append(loss, history.history['loss'][-1])
val_loss = np.append(val_loss, history.history['val_loss'][-1])
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
yhat = yhat * scaler1[0,1] + scaler2[0,1]
#yhat = np.e**yhat - 1
yhat = yhat.round()

y = test_y * scaler1[0,1] + scaler2[0,1]
#y = np.e**y - 1
y = y.round()

y_test = y
y_pred = yhat

#result = np.concatenate((yhat, y), axis = 1)
#np.savetxt(sys.argv[2], result, delimiter=',')

a_pred = np.array([np.round(np.sum(y_pred[0:8784,0])/366), np.round(np.sum(y_pred[8784:17544,0])/365)])
a_test = np.array([np.round(np.sum(y_test[0:8784,0])/366), np.round(np.sum(y_test[8784:17544,0])/365)])
    
AADT_test = np.append(AADT_test,a_test)
AADT_pred = np.append(AADT_pred,a_pred)

hrmse = np.append(hrmse, sqrt(mean_squared_error(y_test, y_pred)))
hmape = np.append(hmape, np.average(np.absolute(y_pred - y_test) / y_test) *100)
armse = np.append(armse, sqrt(mean_squared_error(a_test, a_pred)))
amape = np.append(amape, np.average(np.absolute(a_pred - a_test) / a_test) *100)
    
np.savetxt("lstm.csv", np.column_stack((ATRs,missing_value,loss,val_loss,hrmse,hmape,armse,amape)), fmt='%.3f', delimiter=",")
np.savetxt("lstm_aadt.csv", np.column_stack((AADT_test,AADT_pred)), fmt='%.3f', delimiter=",")

# ahat = np.array([np.round(np.sum(yhat[0:8783,0])/366), np.round(np.sum(yhat[8784:17544,0])/365)])

# a = np.array([np.round(np.sum(y[0:8783,0])/366), np.round(np.sum(y[8784:17544,0])/365)])

# print(a, ahat)

# pyplot.plot(y, label='y')
# pyplot.plot(yhat, label='yhat')
# pyplot.legend()
# pyplot.show()

# rmse = sqrt(mean_squared_error(yhat, y))
# mae = mean_absolute_error(yhat, y)
# mape = np.average(np.absolute(yhat - y) / y) *100
# print('Test RMSE, MAE, MAPE: %.3f, %.3f, %.3f' % (rmse, mae, mape))