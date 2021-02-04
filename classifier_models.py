from data_preprocessing import dataNormalisation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_excel('clinical_dataset.xlsx')

# Normalise the numerical data before training, removes the 'Status' column in pre-processing
x_split = dataNormalisation(df)
y_split = df['Status']

# Shuffle and split the data, shuffle is TRUE by default
x_train, x_test, y_train, y_test = train_test_split(x_split, y_split, test_size=0.10, random_state=13)

# ANN classifier with 2 hidden layers of 50 neurons
classifier = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=100,
                           solver='lbfgs', verbose=1, tol=0.000000001, random_state=13)
classifier.fit(x_train, y_train)

# Calculate accuracy for classifier
score = classifier.score(x_test, y_test)
print("Accuracy of ANN classifier:", score)

# Checking ///
# y_pred = classifier.predict(x_test)
# print(y_pred)
# print(y_test)
