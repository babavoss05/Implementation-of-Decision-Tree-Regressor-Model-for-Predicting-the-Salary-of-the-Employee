# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.



## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: GOKUL
RegisterNumber:  212221220013
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
### data.head():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/82039e99-5a8a-47d0-8541-122dfb8d67cc)

### data.info():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/dbf78d1b-88b4-4e38-933e-4afa3226cce8)

### isnull() & sum() function:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/3289b1e6-af9a-4884-880e-3ce95b2d21c3)

### data.head() for Position:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/27f463c0-a5b1-4af2-8348-a216a8642fef)

### MSE value:
 ![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/79f41538-9405-4180-b2a8-8b2ed7300d7d)

### R2 value:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/29e03ddb-db3e-43dd-9747-c850682e0fcc)

### Prediction value:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/103019882/3dbda533-d2fd-4e30-a8e6-a5f41957a289)







## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
