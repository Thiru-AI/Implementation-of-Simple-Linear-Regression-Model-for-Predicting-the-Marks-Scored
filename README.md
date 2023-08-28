# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thirugnanamoorthi G
RegisterNumber: 212221230117
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)

rmse=np.sqrt(mse)
print("RMSE =",rmse)
```

## Output:
### df.head()
![image](https://user-images.githubusercontent.com/93587823/229069101-af29452a-d961-418d-b366-24130cec4bd2.png)
### df.tail()
![image](https://user-images.githubusercontent.com/93587823/229069686-e8c225b0-9b1c-452c-b42c-420d27f0f3c7.png)
### Array value of X:
![image](https://user-images.githubusercontent.com/93587823/229070072-046735a9-3f1f-4efd-9d11-9f7848460e1f.png)
### Array value of Y:
![image](https://user-images.githubusercontent.com/93587823/229070206-c5fc93d2-9403-4a7b-b617-5e660933845f.png)
### Value of Y prediction:
![image](https://user-images.githubusercontent.com/93587823/229070311-139f4c99-8010-4364-a913-4abb3b93cc2d.png)
### Array values of Y test:
![image](https://user-images.githubusercontent.com/93587823/229070396-5548c7e0-9328-48cf-aa5c-d0539374d689.png)
### Training Set graph:
![image](https://user-images.githubusercontent.com/93587823/229070506-e3cbe6cf-36a3-4dd8-90ac-faa4c7a98b0d.png)
### Test Set graph:
![image](https://user-images.githubusercontent.com/93587823/229070684-459244c0-3c49-448a-b044-56b6c5d6a34e.png)
### Value for MSE, MAE and RMSE:
![image](https://user-images.githubusercontent.com/93587823/229070784-c37ced71-13a2-4325-8c36-a28d92dcdabf.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
