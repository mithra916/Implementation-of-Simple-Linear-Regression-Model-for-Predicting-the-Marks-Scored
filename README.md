# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LOGA MITHRA R
RegisterNumber:  212223100027

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test date
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:

![image](https://github.com/user-attachments/assets/18bbea5d-6747-47f5-9d7f-4a67dca1f2a8)

![image](https://github.com/user-attachments/assets/e2d8a188-abb9-41e7-adf5-f6c862643522)

![image](https://github.com/user-attachments/assets/b1129dd0-5e93-422a-8fd4-221239423285)

![image](https://github.com/user-attachments/assets/b9a1fbfa-2b2f-4450-bfa2-d510fec8726b)

![Uploading image.png…]()

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
