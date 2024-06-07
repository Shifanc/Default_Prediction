# Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics


# Load data
df=pd.read_csv("Default_Fin.csv")
print("DATA :")
print(df.head(10))

# Checking the types of data
print("Types of data")
print(df.info())

# Checking null values
print("Null values")
print(df.isnull().sum())

# Checking for null duplicates
print("Duplicates")
print(df.duplicated().sum())

# Correlation
print("Correlation")
print(df.corr())

#Extracting Independent and dependent Variable
x= df.iloc[:,1:4].values # selecting Employed,Bank Balance,Annual salary
y= df.iloc[:,4].values   #Selecting Defulted

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

# Random forest
# Fitting Decision Tree classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)

# #Predicting the test set result
y_pred= classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())

# from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

























