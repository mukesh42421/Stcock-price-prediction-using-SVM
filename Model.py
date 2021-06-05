# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
dataframe = pd.read_csv("MSFT.csv")

dataframe.info()

dataframe["Date"]=pd.DatetimeIndex(dataframe["Date"])
dataframe.info()

dataframe["Day"] = dataframe["Date"].dt.day
dataframe["Month"] = dataframe["Date"].dt.month
dataframe["Year"] = dataframe["Date"].dt.year
print(dataframe.head())

# spliting into X  and y
X=dataframe[['Day','Month','Year','Open']]

#taking "CLOSE"as target variable
y=dataframe[['Close']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,shuffle=False)



# using SVM
from sklearn.svm import SVR

svr_poly = SVR(kernel="poly",degree=8)
svr_poly.fit(X_train,y_train)

#predicting y
y_pred_poly = svr_poly.predict(X_test)
import pickle
pickle.dump(svr_poly,open('model_close.pkl','wb'))






#taking target variable as "HIGH"

y1 =dataframe[["High"]]

X_train,X_test,y_train,y_test= train_test_split(X,y1,test_size=.3,shuffle=False)

# applying ploy kernal
svr_poly_high = SVR(kernel="poly",degree=8)
svr_poly_high.fit(X_train,y_train)

# predicting
y_pred_high = svr_poly_high.predict(X_test)

#dumpping the file into pickle
pickle.dump(svr_poly_high,open("model_high.pkl","wb"))






#taking target variable as "LOW"

y3=dataframe[["Low"]]
X_train,X_test,y_train,y_test= train_test_split(X,y3,test_size=.3,shuffle=False)

# applying ploy kernal
svr_poly_low = SVR(kernel="poly",degree=8)
svr_poly_low.fit(X_train,y_train)

# predicting
y_pred_low = svr_poly_low.predict(X_test)
pickle.dump(svr_poly_low,open("model_low.pkl","wb"))

# dumping the the file into pickle
pickle.dump(svr_poly_low,open("model_low.pkl","wb"))


