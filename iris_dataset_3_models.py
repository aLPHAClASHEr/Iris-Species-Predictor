import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

iris = datasets.load_iris()
#print(iris.DESCR)

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['labels'] = iris.target

#print(df.head())

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

#print(df.isnull().sum())

#print(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
#X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.25, random_state= 2, shuffle = True, stratify =Y)
#print(X_test.shape, X_train.shape, X.shape)


'''
# Here the Dataset need not be standardized
# Since the values are all near to each other
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
'''




# K-Nearest Neighbors Model

model1 = KNeighborsClassifier(n_neighbors=4)
model1.fit(X_train, Y_train)

X_train_predict1 = model1.predict(X_train)
#print(X_train_predict1)
accuracy1 = accuracy_score(Y_train, X_train_predict1)

print ("\nK-Neighbors Classifier")
print("Training Data Accuracy:", accuracy1*100)


X_test_predict1 = model1.predict(X_test)
test_accuracy1 = accuracy_score(Y_test, X_test_predict1)
print("Test Data Accuracy:", test_accuracy1*100)

# Building a predictive System

input_data1 = (6.7,3.3,5.7,2.5)
#print(type(input_data1))
#print(input_data1)
new_array = np.asarray(input_data1).reshape(1, -1)
#print(new_array)
prediction1 = model1.predict(new_array)
#print(prediction1)
if prediction1[0] == 0:
    print('Iris-Setosa')
elif prediction1[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')
print("\n")






# Logistic Regression Model


model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_predict = model.predict(X_train)
#print(X_train_predict)
accuracy = accuracy_score(Y_train, X_train_predict)

print("Logistic Regression Model")
print("Training Data Accuracy:", accuracy*100)


X_test_predict = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_predict)
print("Test Data Accuracy:", test_accuracy*100, "\n")

# Building a predictive System

input_data = (6.1,2.8,4.0,1.3)
#print(type(input_data))
#print(input_data)
new_array1 = np.asarray(input_data).reshape(1, -1)
#print(new_array1)
prediction = model.predict(new_array1)
#print(prediction)

if prediction[0] == 0:
    print('Iris-Setosa')
elif prediction[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')
    
print("\n")







#Random Forest Model

model2 = RandomForestClassifier()
model2.fit(X_train, Y_train)

X_train_predict2 = model2.predict(X_train)
#print(X_train_predict)
accuracy2 = accuracy_score(Y_train, X_train_predict2)

print("Random Forest Classifier")
print("Training Data Accuracy:", accuracy2*100)


X_test_predict2 = model2.predict(X_test)
test_accuracy2 = accuracy_score(Y_test, X_test_predict2)
print("Test Data Accuracy:", test_accuracy2*100)

# Building a predictive System

input_data2 = (5.7,4.4,1.5,0.4)
new_array2 = np.asarray(input_data2).reshape(1, -1)
prediction2 = model2.predict(new_array2)
if prediction2[0] == 0:
    print('Iris-Setosa')
elif prediction2[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')

print("\n")






# Saving the modell to deploy in streamlit

filename1 = 'knn_model.sav'
loaded_model1 = pickle.dump(model1, open('knn_model.sav', 'wb'))

'''
filename = 'regression_model.sav'
loaded_model = pickle.dump(model, open('regression_model.sav', 'wb'))

filename2 = 'rfclassifier_model.sav'
loaded_model2 = pickle.dump(model2, open('rfclassifier_model.sav', 'wb'))
'''
input_data1 = (4.3,3.0,1.1,0.1)
#print(type(input_data1))
#print(input_data1)
new_array = np.asarray(input_data1).reshape(1, -1)
#print(new_array)
prediction1 = model1.predict(new_array)
#print(prediction1)
if prediction1[0] == 0:
    print('Iris-Setosa')
elif prediction1[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')
print("\n")


input_data = (6.1,2.8,4.0,1.3)
#print(type(input_data))
#print(input_data)
new_array1 = np.asarray(input_data).reshape(1, -1)
#print(new_array1)
prediction = model.predict(new_array1)
#print(prediction)

if prediction[0] == 0:
    print('Iris-Setosa')
elif prediction[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')
    
print("\n")


input_data2 = (5.7,4.4,1.5,0.4)
new_array2 = np.asarray(input_data2).reshape(1, -1)
prediction2 = model2.predict(new_array2)
if prediction2[0] == 0:
    print('Iris-Setosa')
elif prediction2[0] == 1:
    print('Iris-Versicolour')
else:
    print('Iris-Virginica')

print("\n")
