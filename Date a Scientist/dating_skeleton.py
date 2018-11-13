import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:

df = pd.read_csv("profiles.csv")

plt.hist(df.height, bins=12)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.xlim(36, 84)
plt.show()

plt.hist(df.income, bins=20)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.xlim(0, 1000000)
plt.show()

body_type_mapping = {"average": 0, "fit": 1, "athletic": 2, "thin": 3, "curvy": 4, "a little extra": 5, "skinny": 6, "full figured": 7, "overweight": 8, "jacked": 9, "used up": 10, "rather not say": 11}
df["body_type_code"] = df.body_type.map(body_type_mapping)

orientation_mapping = {"straight": 0, "gay": 1, "bisexual": 2}
df["orientation_code"] = df.orientation.map(orientation_mapping)

df = df.fillna(0)

##from data import points, labels
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

##df1 = pd.DataFrame({df3, df4})

list1 = list(zip(df["height"], df["body_type_code"]))

training_data, validation_data, training_labels, validation_labels = train_test_split(list1, df["orientation_code"], train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = 'rbf', gamma = 0.1)

classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(list1, df["orientation_code"])
test1 = (classifier.predict(list1))

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = df["orientation_code"]
guesses = test1

print(accuracy_score(labels, guesses))
##print(recall_score(labels, guesses))
##print(precision_score(labels, guesses))
##print(f1_score(labels, guesses))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

list2 = list(zip(df["age"],df["height"]))

x_train, x_test, y_train, y_test = train_test_split(list2, df["income"], train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.plot(range(12), range(12))

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()

##from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(list2,df["income"])
test2 = regressor.predict(list2)

test3 = test2.reshape(-1,1)

print(test3)

print(regressor.score(df["income"].values.reshape(-1,1), test3))
