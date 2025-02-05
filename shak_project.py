import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
store_data = pd.read_csv('store.csv')
store_data.drop(['RetailType','RivalOpeningMonth','RivalEntryYear','ContinuousBogoMonths'],axis=1,inplace=True)
training_data = pd.read_csv('train.csv')
Data = pd.merge(training_data, store_data, on='Store_id')
Data.drop('NumberOfCustomers',axis=1,inplace=True)
median_distance = store_data['DistanceToRivalStore'].median() 
store_data['DistanceToRivalStore'] = store_data['DistanceToRivalStore'].fillna(median_distance)
Data.fillna(0, inplace=True)
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Year'] = Data['Date'].dt.year
Data['Month'] = Data['Date'].dt.month
Data['Day'] = Data['Date'].dt.day
Data['WeekOfYear'] = Data['Date'].dt.isocalendar().week
Data.sort_values(by='Date', inplace=True)
Data.drop('Date', axis=1, inplace=True)
mapping = {'a':1,'b':2,'c':3}
Data['Stock variety'] = Data['Stock variety'].map(mapping)
standardizer = StandardScaler()
standardizer.fit_transform(Data)
X_train, X_test, y_train, y_test = train_test_split(Data.drop('Sales', axis=1), Data['Sales'], test_size=0.3,shuffle=False)
linear_mod = LinearRegression()
random_forest = RandomForestRegressor(n_estimators=100)
linear_mod.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
'''print('Linear Regression accuracy:',mean_squared_error(y_test, linear_mod.predict(X_test)))
print('Random Forest accuracy:', mean_squared_error(y_test, random_forest.predict(X_test)))'''
print(Data.drop('Sales').columns.tolist()[np.unravel_index(np.argmax(random_forest.feature_importances_),random_forest.feature_importances_.shape)[0]])