import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel('clinical_dataset.xlsx')

y_split = df['Status']
x_split = df.drop(['Status'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_split, y_split, test_size=0.10, random_state=13)

cv = ShuffleSplit(n_splits=10)  # 10 fold Cross-validation initialise

# ANN classifier with 2 hidden layers of 50 neurons
ANN_Classifier50 = MLPClassifier(hidden_layer_sizes=(50, 50), activation='logistic', solver='lbfgs', verbose=1,
                                 tol=0.000000001, random_state=13)

# ANN classifier with 2 hidden layers of 500 neurons
ANN_Classifier500 = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', solver='lbfgs', verbose=1,
                                  tol=0.000000001, random_state=13)

# ANN classifier with 2 hidden layers of 100 neurons
ANN_Classifier1000 = MLPClassifier(hidden_layer_sizes=(1000, 1000), activation='logistic', solver='lbfgs', verbose=1,
                                   tol=0.000000001, random_state=13)

# Calculate the mean accuracy over 10 iterations
ANN_scores50 = cross_val_score(ANN_Classifier50, x_split, y_split, cv=cv)
ANN_scores500 = cross_val_score(ANN_Classifier500, x_split, y_split, cv=cv)
ANN_scores1000 = cross_val_score(ANN_Classifier1000, x_split, y_split, cv=cv)

# Random forest classifier with 20 trees
RF_classifier20 = RandomForestClassifier(n_estimators=20, min_samples_leaf=5, verbose=1, random_state=13)

# Random forest classifier with 500 trees
RF_classifier500 = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, verbose=1, random_state=13)

# Random forest classifier with 10000 trees
RF_classifier10000 = RandomForestClassifier(n_estimators=10000, min_samples_leaf=5, verbose=1, random_state=13)

# Calculate the mean accuracy over 10 iterations
RF_scores20 = cross_val_score(RF_classifier20, x_split, y_split, cv=cv)
RF_scores500 = cross_val_score(RF_classifier500, x_split, y_split, cv=cv)
RF_scores10000 = cross_val_score(RF_classifier10000, x_split, y_split, cv=cv)

# Print the results
print("Mean accuracy for ANN classifier with 50 neurons over 10-fold CV:", ANN_scores50.mean())
print("Mean accuracy for ANN classifier with 500 neurons over 10-fold CV:", ANN_scores500.mean())
print("Mean accuracy for ANN classifier with 1000 neurons over 10-fold CV:", ANN_scores1000.mean())
print("------------------------------------------------------------------------------------------------")
print("Mean accuracy for random forest classifier with 20 trees over 10-fold CV:", RF_scores20.mean())
print("Mean accuracy for random forest classifier with 500 trees over 10-fold CV:", RF_scores500.mean())
print("Mean accuracy for random forest classifier with 10000 trees over 10-fold CV:", RF_scores10000.mean())
