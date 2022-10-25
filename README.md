
# Machine Learning - Machine Learning - Cancer Classification (CMP3751M_Assessment_02)


## Overview

A program that contains a artificial neural network (ANN) and a random forest classifier model for automatically classifying patients as healthy or cancerous, based on clinical features from cancer screenings.


## Dataset Information

The dataset has two classes (healthy/cancerous condition) and 9 clinical features, including age, body mass index (BMI), glucose, insulin, leptin, resistin, MCP.1, HOMA, and adiponectin. The class membership of each row is stored in the field “Status”. Status refers to the health condition of patients, this will be used as our label/annotation for the sake of all implementations. Unit of measurement or range of values of each feature are not relevant. However, features can be at different scales and/or measured in different units.
   
   
## Summary of Implementation

### Data Import and Pre-Processing

The dataset containing two classes (healthy/cancerous conditions) and nine clinical features such as, age, BMI, glucose, and others was a excel file of the format ‘.xlsx’, This dataset was read into the python IDE using the pandas ‘.read_excel’ function and stored as a Pandas DataFrame, to allow for efficient data analysis and manipulation.

Pre-processing is the first step taken in many data analysis techniques. It includes data cleaning, normalization, transformation, feature extraction, and selection, etc. (Kotsiantis, Kanellopoulos and Pintelas, 2006). Meaning the data is being prepared appropriately to be used in a later stage. The outputted size of the dataset is printed to the console, showing: the total number of elements present, and a list of all features included as well as the number of missing features present. The size of the dataset was simply found by using the ‘len’ function on the Dataframe to find the number of rows and again ‘len’ being used in the ‘DataFrame.columns’ variable to find the number of columns. In order to report which features were found in the dataset, the ‘.columns.values’ function was used, which passes a list of all the headers to a variable. The final column (Status) was then removed as it is not a feature, and this variable is passed to another function that converts the list to a string with ‘/’ as the delimiter and outputs this to the console.

Categorical variables are usually those that can only take on a fixed number of possible values, such as genders, race, blood types, etc. The only categorical variable present in the dataset was the status of each patient, as it could only be either healthy or cancerous. Next, the data was normalised before any training could be performed. Data normalisation is the process that involves making sure all the data in the dataset is spread equally; doing so eliminates redundancy and increases the integrity of the data. Normalising the data between the values of zero and one for our dataset is crucial to obtaining good results, as well as to significantly fasten the calculations (Sevilla, 1997). However, data only is required to be normalised when the features have different ranges of values, so this process is not
always required, but for this dataset, it was found to be beneficial.

The method chosen for normalisation was the min-max method. Performing this calculation results in all the data being transformed between zero and one. Firstly, the status column is excluded from the dataset as this is a categorical variable and cannot be normalised. The new dataset variable ‘statusDropped’ is then passed into the formula, and the resulting normalised data is calculated and returned. This data normalisation function is the called, and the resulting dataset can be seen in the console output as well as being stored for later training.


### Designing the Algorithms

Initially, the data is separated into x and y sections. The x values are those of the entire numerical dataset, such as age, BMI, glucose, etc., and the y values are an array of the patient’s statues. The numerical data (x) is then passed to the normalisation function to get all the numbers spread across and equal range between 0 and 1. Next, the method used to split and shuffle the data was the ‘train_test_split’ function from the scikit-learn machine learning library in python. Both sets of split data are passed in as input arguments, as well as a decimal value for the desired size of the test set, which was required to be 10% leaving 90% of the data for training. This means there are 13 rows left for the testing dataset. The function, by default, ensures the data is randomly shuffled but the values stay consist across rows. The x and y data for both training and testing is then returned and can be used by the models later.

#### Artifical Neural Network (ANN)

The artificial neural network (ANN) model uses the ‘MLPClassifier’, which is imported from the ‘sklearn.neural_network’ sub-library. This is a multi- layer perceptron classifier and can be highly modified using a variety of input arguments. Firstly, the number of neurons is set in each hidden layer to 500, using the ‘hidden_layer_size’. As the architecture requires two hidden layers, 500 is repeated here twice for each layer. Next, the activation function is set to logistic as this represents the logistic sigmoid function required by the brief. The sigmoid function is used as the values only exist between the values of 1 and 0, therefore it is useful in our model as we are only trying to predict between two variables, healthy or cancerous. It is plotted as an S-shaped curve starting at -1 and the formula can be seen in Figure 12. The ‘max_iter’ variable is then set to be equal to 100, which is the number of iterations the model will undertake. Through the process of trail and error, 100 iterations were found to be a good value for receiving an acceptable accuracy score while reducing the compilation time of the model. Despite increasing the iterations above this value and subsequently increasing the processing time there was very little gain found in the accuracy, only a few percent. Following this, the solver for the ANN classifier was set the ‘lbfgs’ which stands for limited-memory- Broyden-Fletcher-Goldfarb-Shanno. There are three solvers that classifier model can use, with the default being ‘adam’, which is typically preferable for relatively large datasets. However, it was found through comparing the resultant accuracy of each of these solvers that ‘lbfgs’ had on average the best accuracy. This solver approximates the second derivative matrix updates with gradient evaluations (Hale, 2020). Nevertheless, it only stores the last few updates so is saved in memory, meaning it isn’t ideal for large datasets, but for our dataset works sufficiently. Finally, the verbose is set to one or true, which simply displays the progress training and the ‘random_state’ argument is given the seed of 13 to ensure reproducible results.

The created ANN model is then passed to the ‘ANN_classifier’ variable, which is given the ‘.fit’ method and fits the model to the dataset. Additionally, the accuracy was found by passing the newly created model with the ‘.score’ method and the separate testing dataset. Doing so would have the model look at the numerical data from the test set, which has never been seen by the model before, and it would predict the status of the patient based on this data. These outputted class labels are then compared to the actual correct result to give a mean accuracy value based on how many were correctly classified. In this integration of the code, an accuracy of 92.31% was found. As there are only 13 possible correct classes to be predicted, having a mean accuracy this high is ideal.

#### Random Forest Classifier

To train the random forest classifier, again the scikit-learn based library ‘sklearn.ensemble’ is imported and the ‘RandomForestClassifier’ function is used. The number of trees required is 1000, which is set by the ‘n_estimator’ argument and as both, 5 and 50 minimum number of samples at each leaf node is required, this can be set by the ‘min_samples_leaf’ argument when called. The majority of the other input arguments are kept at default for this model as they are all already optimal. Again, this is passed to the ‘.fit’ function with the training data and the model begins training. As there are only 1000 trees for both these models, training doesn’t require much time.

In order to find the accuracy for each random forest model a different method was used compared to the ANN model. The model was passed to the ‘.predict’ function alongside the numerical table data (x values), which would predict the class labels of either healthy or cancerous just as the ‘.score’ function would. However, these outputted labels were stored in a variable called ‘RF_pred’. These labels are compared to the actual labels stored in ‘y_test’ using the ‘accuracy_score’ function, which would output a decimal accuracy score. Following this, the result is displayed to the console using the print function which also converted the decimal value to a percentage for easier viewing. The above function was called twice, using the same data values but different numbers for the minimum numbers of samples required at each leaf node. Figure 15 shows this output and it can be clearly seen that the random forest classifier with 5 samples has a much higher accuracy than on having 50 minimum samples.

Furthermore, another function was created for the random forest models. This function allowed for the number of trees to be altered. The minimum number of leaf samples for all these results is set to 5 and the rest of the parameters are kept at their default values. It can be seen that the highest accuracy (in a previous test) was again, 84.62% which was the classifier with 1000 trees. The model with the least number of trees starts with the lowest accuracy and as the amount increase, so does the accuracy. However, at a certain point, above 1000 trees, the accuracy starts to decrease again. This may be due to the model overfitting towards the training data and causing a loss in accuracy and performance.

### Model Evaluation

Firstly, the 10-fold CV method is initialised using the ‘ShuffleSplit’ function from the ‘sklearn.model_selection’ sub-library. The 10 folds are defined by the ‘n_splits’ variable, which will split the data into 10 folds of almost equal sizes. After this has been defined, the data is split into x and y variables, but no other pre-processing is performed.

The model’s parameters are created and defined as they were before with differing numbers of neurons in each hidden layer for the ANN and different numbers of trees for the random forest classifiers. For the random forest classifier, the minimum number of samples required to be at a leaf node was set to 5, as from previous evaluation this was found to be the superior value in guaranteeing higher accuracy. For all the models the ‘cross_val_score’ function was used to evaluate the score of the cross-validation. This was a built-in function from scikit-learn and takes the input arguments of the model, data, and the predetermined number of k-folds.

To get an average and final accuracy for each model, to allow for comparison, the mean of this array was found, simply using the ‘.mean’ function and the results printed to the console.


## References


Kotsiantis, S., Kanellopoulos, D. and Pintelas, P.E. (2006). Data Preprocessing for Supervised Leaning. International Journal of Computer Science, [online] 1(1), pp.111–117. Available at: https://www.researchgate.net/publication/228084519_Data_Preprocessing_for_Supervised_Learning [Accessed 24 Jan. 2021].

Sevilla, J. (1997). Importance of input data normalization for the application of neural networks to complex industrial problems. [online] ResearchGate. Available at: https://www.researchgate.net/publication/3135573_Importance_of_input_data_normalization_for_the_application_of_neural_networks_to_complex_industrial_problems [Accessed 26 Jan. 2021].

