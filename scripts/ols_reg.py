import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from predictive_imputer import predictive_imputer
import impyute as impy
from matplotlib import pyplot as plt

ATRs = np.delete(np.arange(1,150),obj=[24,26,28,29,45,50,52,61,77,78,100,137,138,145])
#ATRs = np.arange(1,150)
#print(ATRs)

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

    # Model:Regression
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 1)
    X_train = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_train, y_train)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    X_test = poly_reg.fit_transform(X_test)
    y_pred = regressor.predict(X_test)
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

np.savetxt("regression.csv", np.column_stack((ATRs,missing_value,hrmse,hmape,armse,amape)), fmt='%.3f', delimiter=",")
np.savetxt("reg_aadt.csv", np.column_stack((AADT_test,AADT_pred)), fmt='%.3f', delimiter=",")
