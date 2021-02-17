import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Read in the data as a pandas dataframe
def dataImport(filename):
    data = pd.read_excel(filename)
    return data


# Normalise the data using a min-max normalisation method
def dataNormalisation(dataset):
    statusDropped = dataset.drop(['Status'], axis=1)
    normalised = (statusDropped - statusDropped.min()) / (statusDropped.max() - statusDropped.min())
    return normalised


# Split and shuffle the data into x and y for training and testing
def dataPreprocessing(dataset):
    # Normalise the numerical data before training, removes the 'Status' column in 'dataNormalisation'
    x_split = dataNormalisation(dataset)
    y_split = dataset['Status']

    # Shuffle and split the data, shuffle is TRUE by default
    x_train, x_test, y_train, y_test = train_test_split(x_split, y_split, test_size=0.10)
    return x_train, x_test, y_train, y_test


# NOTE:
# An iteration (max_iter) is a gradient update step
# An epoch is a pass over the entire dataset
# https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks

# ANN classifier with 2 hidden layers of 500 neurons each
def ANN_classifier(x_train, x_test, y_train, y_test):
    ANN_classifier = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=100,  # CHANGE ITER?
                                   solver='lbfgs', verbose=1, random_state=13)
    ANN_classifier.fit(x_train, y_train)

    # Calculate and report accuracy for ANN classifier
    ANN_score = ANN_classifier.score(x_test, y_test)
    print('--------------------------------------------------------------------------------------')
    print('Accuracy of ANN classifier with 2 hidden layers of 500 neurons each: ', round((ANN_score * 100), 2), '%',
          sep='')
    print('--------------------------------------------------------------------------------------')
    return


# Random forest classifier with 1000 trees
def RF_classifier(x_train, x_test, y_train, y_test, min_samples_leaf):
    RF_classifier = RandomForestClassifier(n_estimators=1000, min_samples_leaf=min_samples_leaf)
    RF_classifier.fit(x_train, y_train)
    RF_pred = RF_classifier.predict(x_test)

    # Calculate and report accuracy for the random forest classifier
    RF_score = metrics.accuracy_score(y_test, RF_pred)
    print('Accuracy of random forest classifier with a minimum of ', min_samples_leaf, ' samples at a leaf node: ',
          round((RF_score * 100), 2), '%', sep='')
    return


# Random forest classifier for variable number of trees
def RF_tweak(numTrees):
    RF_classifier = RandomForestClassifier(n_estimators=numTrees, min_samples_leaf=5)
    RF_classifier.fit(x_train, y_train)
    RF_pred = RF_classifier.predict(x_test)

    # Calculate and report accuracy for the random forest classifier
    RF_score = metrics.accuracy_score(y_test, RF_pred)
    print('Accuracy of random forest classifier with ', numTrees, ' trees: ', round((RF_score * 100), 2), '%', sep='')
    return


if __name__ == '__main__':
    # Preparing data for training
    df = dataImport('clinical_dataset.xlsx')
    x_train, x_test, y_train, y_test = dataPreprocessing(df)

    # ANN classifier with ANN classifier with 2 hidden layers of 500 neurons each
    ANN_classifier(x_train, x_test, y_train, y_test)

    # Random forest classifier with 1000 trees and 5 minimum samples at a leaf node
    RF_classifier(x_train, x_test, y_train, y_test, 5)
    # Random forest classifier with 1000 trees and 50 minimum samples at a leaf node
    RF_classifier(x_train, x_test, y_train, y_test, 50)

    # Tweaking the number of trees to see performance changes
    print('--------------------------------------------------------------------------------------')
    RF_tweak(10)
    RF_tweak(50)
    RF_tweak(100)
    RF_tweak(1000)
    RF_tweak(2500)
    RF_tweak(5000)
    print('--------------------------------------------------------------------------------------')
