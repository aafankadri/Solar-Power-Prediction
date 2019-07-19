import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#file input
dataset = pd.read_csv('11MW-GenerationFile-5Min - Copy.csv')
x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 7].values

#missing data
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])'''

#Categrial veriable
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

#spliting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting
y_pred = regressor.predict(x_test)

#visualising
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue' )
plt.title('salary vs exp (Training set)')
plt.xlabel('years of exp')
plt.ylabel('salary')

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue' )
plt.title('salary vs exp (test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')