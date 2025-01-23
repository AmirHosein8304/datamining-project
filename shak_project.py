import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
store_data = pd.read_csv('store.csv')
store_data.drop(['RetailType','RivalOpeningMonth','RivalEntryYear'],axis=1,inplace=True)
training_data = pd.read_csv('train.csv')
Data = pd.merge(training_data, store_data, on='Store_id')
Data.drop('NumberOfCustomers',axis=1,inplace=True)
Data['DistanceToRivalStore'].fillna(Data['DistanceToRivalStore'].median(), inplace=True)
Data.fillna(0, inplace=True)
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Year'] = Data['Date'].dt.year
Data['Month'] = Data['Date'].dt.month
Data['Day'] = Data['Date'].dt.day
Data['WeekOfYear'] = Data['Date'].dt.isocalendar().week
standardizer = StandardScaler()
standardizer.fit_transform(Data.drop(['Stock variety','ContinuousBogoMonths','Date'], axis=1))
X_train, X_test, y_train, y_test = train_test_split(Data.drop('Sales', axis=1), Data['Sales'], test_size=0.2, random_state=42)
