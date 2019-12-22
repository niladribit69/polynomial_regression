
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting a linear regressor
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X,y)
lpred = lreg.predict(X)

# Fitting a polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
preg = PolynomialFeatures(degree = 4)
X_poly = preg.fit_transform(X)
lpreg = LinearRegression()
lpreg.fit(X_poly,y)
ppred = lpreg.predict(X_poly)

sample_out = lpreg.predict(preg.fit_transform([[6.5]]))


# Visualizing data
plt.scatter(X,y,color='red')
plt.plot(X,ppred, color = 'blue')
plt.show()




