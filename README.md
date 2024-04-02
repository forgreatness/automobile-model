# Automobile Model

## Description
In this project, we will be working with a dataset from the UCI AI learning repository. The dataset will be used to create a model to predict the price of a vehicle. Which makes this model a regression. There are many different types of mathematical concepts used to predict and get numbers such as: knn, ann, svm but we will use KNN for simplicity

## Dataset
From: 1985 Ward's Automotive Yearbook
** entries **: 205
** features **: 25
predicted Value: price


## Concept
I. Data Preprocessing
- A. Read the data into the program from the dataset file
- B. Make sure the data when read in have labels for both the columns and rows so that they can be process easier
- C. Choose only the relevant features you think its necessary which are the columns needed
- D. Normalize the data to optimize the performance
- E. Make sure that the data you have do not include any NAN(not a number) or missing values (null, empty)
II. Configuring the model
- A. Choose the K nearest neighbors by trial
- B. Divide the data into training set, test set, and validation as well
- C. Try different values for nearest_neighbors from 1-25
- D. Calculate the rooted mean square error of each k values
- E. Repeat step D for each feature
- F. Repeat step D for each range of feature
III. Plot the data
- A. For step E of part II. Plot each feauture along the X axis and the mean of the k-nearest neighbor among the y to see which one is best
- B. For step F of part II. Plot each k-values along the X axis, and the average rooted mean square error for that k-value along the y to see which range of features is best

## Results
The overal results discovered is that having a small k-values around 2 and a small number of features to test had the lowest mean squared error.
