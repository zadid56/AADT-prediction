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
from matplotlib import pyplot as plt

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

ATRs = np.array([4])
#ATRs = np.delete(np.arange(1,150),obj=[24,26,28,29,45,50,52,61,77,78,100,137,138,145])
hours_in_day = 8736
AADT_test = []
AADT_pred = []

loss = []
val_loss = []
hrmse = []
hmape = []
armse = []
amape = []
missing_value = []

for i in ATRs:
    dataset = read_csv('/home/mdzadik/pythonopt/ATR_data/'+str(i)+'.csv', usecols=range(0,2))
    values = dataset.values
    values = values.astype('float')
    values2 = []
    values2 = np.append(values2,values[0:78912,0])
    values2 = np.append(values2,values[70152:78912,1])
    values2 = values2.reshape(87672,1)

    missing_value = np.append(missing_value, (len(values2)-np.count_nonzero(values2))*100/len(values2))

    values[values == 0] = np.nan
    #imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
    #values = imputer.fit(values).transform(values.copy())
    values = impy.median(values)
    values[values < 0] = 0
    
    y_train = values[0:61368,1].reshape(61368,1)
    y_test = values[61368:78912,1].reshape(17544,1)
    
    n_train = 61368 - hours_in_day
    n_test = 17544
    
    #a1 = difference(values[:,0], hours_in_day)
    a1 = values[:,0]
    a1 = a1.reshape(len(a1),1)
    a1t = series_to_supervised(a1, n_in=hours_in_day, n_out=1, dropnan=True)
    print(np.shape(a1t))
    
    #a2 = difference(values[:,1], hours_in_day)
    a2 = values[hours_in_day:,1]
    a2 = a2.reshape(len(a2),1)
    print(np.shape(a2))
    #values = np.concatenate([a1, a2], axis=1)
    values = np.concatenate([a1t, a2], axis=1)
    print(np.shape(values))
    
    #plt.plot(values[:,0],values[:,1])
    #plt.show()

    #values = values.reshape(values.shape[0],2)

    n_row = values.shape[0]
    n_col = values.shape[1]

    train = values[:n_train, :]
    test = values[n_train:, :]

    #train = np.square(train)
    #test = np.square(test)
    
    #train = np.log(1+train)
    #test = np.log(1+test)

    scaler1 = np.amax(train, axis=0) - np.amin(train, axis=0)
    scaler2 = np.amin(train, axis=0)
    #print(scaler1,scaler2)
    scaler1 = scaler1.reshape(1,n_col)
    scaler2 = scaler2.reshape(1,n_col)
    train = (train - scaler2) / scaler1
    test = (test - scaler2) / scaler1
    train = train.astype('float')
    test = test.astype('float')
    #print(scaler1, scaler2)

    # split into input and outputs
    train_X, train_y = train[:, 0:n_col-2], train[:, n_col-1]
    test_X, test_y = test[:, 0:n_col-2], test[:, n_col-1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, n_col-2))
    test_X = test_X.reshape((test_X.shape[0], 1, n_col-2))
    train_y = train_y.reshape((train_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=20, batch_size=8760, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    loss = np.append(loss, history.history['loss'][-1])
    val_loss = np.append(val_loss, history.history['val_loss'][-1])
    #print(loss, val_loss)
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    yhat = yhat * scaler1[0,1] + scaler2[0,1]
    #yhat = np.sqrt(yhat)
    #yhat = np.e**yhat - 1
    yhat = yhat.round()

    y = test_y * scaler1[0,1] + scaler2[0,1]
    #y = np.sqrt(y)
    #y = np.e**y - 1
    y = y.round()

    #y_pred = yhat
    #y_test = y
    forecast = yhat
    
    history2 = [x for x in y_train]
    predictions = []
    hour = 1
    for yhat in forecast:
        inverted = inverse_difference(history2, yhat, hours_in_day)
        #print('hour %d: %f' % (hour, inverted))
        history2.append(inverted)
        predictions.append(inverted)
        hour += 1
    
    y_pred = np.asarray(predictions).reshape(17544,1)
    #plt.plot(y_test,'b')
    #plt.plot(y_pred,'r')
    #plt.show()
    # Results
    
    #result = np.concatenate((y_pred, y_test), axis = 1)
    #np.savetxt("results.csv", result, delimiter=',')
    
    a_pred = np.array([np.round(np.sum(y_pred[0:8784,0])/366), np.round(np.sum(y_pred[8784:17544,0])/365)])
    a_test = np.array([np.round(np.sum(y_test[0:8784,0])/366), np.round(np.sum(y_test[8784:17544,0])/365)])
    
    AADT_test = np.append(AADT_test,a_test)
    AADT_pred = np.append(AADT_pred,a_pred)

    hrmse = np.append(hrmse, sqrt(mean_squared_error(y_test, y_pred)))
    hmape = np.append(hmape, np.average(np.absolute(y_pred - y_test) / y_test) *100)
    armse = np.append(armse, sqrt(mean_squared_error(a_test, a_pred)))
    amape = np.append(amape, np.average(np.absolute(a_pred - a_test) / a_test) *100)
    print(i)
    
np.savetxt("lstm.csv", np.column_stack((ATRs,missing_value,loss,val_loss,hrmse,hmape,armse,amape)), fmt='%.3f', delimiter=",")
np.savetxt("lstm_aadt.csv", np.column_stack((AADT_test,AADT_pred)), fmt='%.3f', delimiter=",")