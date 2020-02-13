import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from predictive_imputer import predictive_imputer
import impyute as impy
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt

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

hours_in_day = 8736

ATRs = np.delete(np.arange(1,150),obj=[24,26,28,29,45,50,52,61,77,78,100,137,138,145])
#ATRs = np.arange(111,136)

hrmse = []
hmape = []
armse = []
amape = []
missing_value = []

AADT_test = []
AADT_pred = []

for i in ATRs:
    dataset = pd.read_csv('/home/mdzadik/pythonopt/ATR_data/'+str(i)+'.csv', usecols=range(0,2))
    values = dataset.values
    values = values.astype('float')
    values2 = []
    values2 = np.append(values2,values[0:78912,0])
    values2 = np.append(values2,values[70152:78912,1])
    values2 = values2.reshape(87672,1)
    
    hrs = np.arange(1,87673)
    hrs = hrs.reshape(87672,1)
    values = np.concatenate([hrs, values2], axis=1)

    missing_value = np.append(missing_value, (len(values2)-np.count_nonzero(values2))*100/len(values2))

    values[values == 0] = np.nan
    #imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
    #values = imputer.fit(values).transform(values.copy())
    values = impy.median(values)
    values[values < 0] = 0
    
    #plt.plot(values[:,0],values[:,1])
    #plt.show()

    X_train = values[0:70128,0].reshape(70128,1)
    y_train = values[0:70128,1].reshape(70128,1)
    X_test = values[70128:87672,0].reshape(17544,1)
    y_test = values[70128:87672,1].reshape(17544,1)

    # Model:HWES
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    differenced = difference(y_train, hours_in_day)
    regressor.fit(X_train[hours_in_day:], differenced)
    #differenced = y_train
    #model = ExponentialSmoothing(differenced, trend='add', seasonal='add', seasonal_periods=8)
    #model = SimpleExpSmoothing(differenced)
    #model_fit = model.fit()
    # multi-step out-of-sample forecast
    forecast = regressor.predict(X_test)
    # invert the differenced forecast to something usable
    history = [x for x in y_train]
    predictions = []
    #predictions = forecast
    hour = 1
    for yhat in forecast:
        inverted = np.round(inverse_difference(history, yhat, hours_in_day))
        #print('hour %d: %f' % (hour, inverted))
        history.append(inverted)
        predictions.append(inverted)
        hour += 1
    
    y_pred = np.asarray(predictions).reshape(17544,1)
    #plt.plot(y_test,'b')
    #plt.plot(y_pred,'r')
    #plt.show()
    # Results
    a_pred = np.array([np.round(np.sum(y_pred[0:8784,0])/366), np.round(np.sum(y_pred[8784:17544,0])/365)])
    a_test = np.array([np.round(np.sum(y_test[0:8784,0])/366), np.round(np.sum(y_test[8784:17544,0])/365)])
    
    AADT_test = np.append(AADT_test,a_test)
    AADT_pred = np.append(AADT_pred,a_pred)

    hrmse = np.append(hrmse, math.sqrt(mean_squared_error(y_test, y_pred)))
    hmape = np.append(hmape, np.average(np.absolute(y_pred - y_test) / y_test) *100)
    armse = np.append(armse, math.sqrt(mean_squared_error(a_test, a_pred)))
    amape = np.append(amape, np.average(np.absolute(a_pred - a_test) / a_test) *100)
    print(i)

np.savetxt("hes.csv", np.column_stack((ATRs,missing_value,hrmse,hmape,armse,amape)), fmt='%.3f', delimiter=",")
np.savetxt("hes_aadt.csv", np.column_stack((AADT_test,AADT_pred)), fmt='%.3f', delimiter=",")
