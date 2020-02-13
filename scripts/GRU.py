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
from keras.layers import GRU
import numpy as np
from predictive_imputer import predictive_imputer
import impyute as impy

# convert series to supervised learning
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
 

dataset = read_csv('1.csv', usecols=range(0,24))
values = dataset.values
values = values.astype('float')

values[values == 0] = np.nan
#imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
#values_imp = imputer.fit(values).transform(values.copy())
values_imp = impy.mean(values)
#values_imp = values
 
n_lag = 730
n_lead = 365
n_train = 1825
n_test = 734
hours = 24

y_bar = np.zeros([n_test,hours])
y_act = np.zeros([n_test,hours])

for i in range(0, hours):
	values = values_imp[:,i]
	values = values.reshape(values.shape[0],1)
	# frame as supervised learning
	reframed = series_to_supervised(values, n_lag, n_lead)

	n_col = reframed.shape[1]
	# split into train and test sets
	values = reframed.values
	print(values.shape)
	temp1 = np.amax(values, axis=0) - np.amin(values, axis=0)
	temp2 = np.amin(values, axis=0)
	scaler1 = temp1.reshape(1,n_col)
	scaler2 = temp2.reshape(1,n_col)
	values = (values - scaler2) / scaler1
	values = values.astype('float32')

	train = values[:n_train, :]
	test = values[n_train:, :]
	# split into input and outputs
	train_X, train_y = train[:, 0:int(n_col*n_lag/(n_lag+n_lead))], train[:, int(n_col*n_lag/(n_lag+n_lead)):int(n_col)]
	test_X, test_y = test[:, 0:int(n_col*n_lag/(n_lag+n_lead))], test[:, int(n_col*n_lag/(n_lag+n_lead)):int(n_col)]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
	n_out = int(train_y.shape[1])

	# design network
	model = Sequential()
	model.add(GRU(n_out, input_shape=(train_X.shape[1], train_X.shape[2])))
	#model.add(Dropout(0.4))
	model.add(Dense(n_out))
	model.compile(loss='mean_squared_error', optimizer='Adam')
	# fit network
	history = model.fit(train_X, train_y, epochs=1500, batch_size=365, validation_data=(test_X, test_y), verbose=0, shuffle=False)
	# plot history
	#pyplot.plot(history.history['loss'], label='train')
	#pyplot.plot(history.history['val_loss'], label='test')
	#pyplot.legend()
	#pyplot.show()

	# make a prediction
	yhat = model.predict(test_X)
	yhat = yhat * scaler1[0,int(n_col*n_lag/(n_lag+n_lead)):int(n_col)] + scaler2[0,int(n_col*n_lag/(n_lag+n_lead)):int(n_col)]
	yhat = yhat.round()
	yhat = np.average(yhat, axis=1)
	yhat = np.round(yhat)
	yhat = yhat.reshape(yhat.shape[0],1)
	y_bar[:,i] = yhat[:,0]

	y = test_y * scaler1[0,int(n_col*n_lag/(n_lag+n_lead)):int(n_col)] + scaler2[0,int(n_col*n_lag/(n_lag+n_lead)):int(n_col)]
	y = y.round()
	y = np.average(y, axis=1)
	y = np.round(y)
	y = y.reshape(y.shape[0],1)
	y_act[:,i] = y[:,0]

y_bar = np.sum(y_bar, axis=1)
y_act = np.sum(y_act, axis=1)

rmse = sqrt(mean_squared_error(y_bar, y_act))
mae = mean_absolute_error(y_bar, y_act)
mape = np.average(np.absolute(y_bar - y_act) / y_act)*100
print('Test RMSE, MAE, MAPE: %.3f, %.3f, %.3f' % (rmse, mae, mape))