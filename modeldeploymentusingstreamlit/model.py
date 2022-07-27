import numpy as np
import pandas as pd
import seaborn as sb
import pickle

data=pd.read_csv("C:/true_car_listings.csv")

print(data.head())

sb.pairplot(data)

print(data.corr())

X=pd.DataFrame({'Year':data['Year'],'Mileage':data['Mileage']})
y=np.array(data['Price'])

print(X)

print(X.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

from sklearn import metrics
print('R2 of Linear Regression:',(metrics.r2_score(y_test,y_pred)))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred),2))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),2))
print('Root Mean Squared Error:',round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))

a=regressor.predict(np.array([2015,20000]).reshape(1, -1))
print(a)

pickle.dump(regressor,open("model.pkl","wb"))
