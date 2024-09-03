import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# importing for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the csv data
df = pd.read_csv('IRIS.csv')
# print(df.head())

# species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
# colour = ['red', 'green', 'blue']

# for i in range(3):
#     x = df[df['species'] == species[i]]
#     plt.scatter(x['sepal_length'], x['sepal_width'], c = colour[i], label=species[i])
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.legend()

# for i in range(3):
#     x = df[df['species'] == species[i]]
#     plt.scatter(x['petal_length'], x['petal_width'], c = colour[i], label=species[i])
# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.legend()

# plt.show()

# display correlation matrix
# df = df.drop(columns=['species'])
# corr = df.corr()
# fig, ax = plt.subplots(figsize=(5,4))
# sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
# plt.show()

# Model Training and Testing
X = df.drop(columns=['species'])
Y = df['species']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model=LogisticRegression()
model.fit(x_train, y_train)
print("Logistic Regression Accuracy: ", model.score(x_test, y_test)*100)

# model = KNeighborsClassifier()
# model.fit(x_train.values, y_train.values)
# print("K-nearest neighbours accuracy: ", model.score(x_test, y_test)*100)

# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)
# print("Decision Tree Accuracy: ", model.score(x_test, y_test)*100)

filename = "saved_model.sav"
pickle.dump(model, open(filename, 'wb'))

print(model.predict([[6,2.2,4,1]]))
print(x_test.head())