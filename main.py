import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 99

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)

# store all the numerical cols in a list[]
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
carsWithNumericalInfo = cars[continuous_values_cols] #carsWithNumericalInfo is a dataframe that contains data of the numerical columns stated above with only these columns as the data

"""
Below we will practice the mos used methods in pandas
[size, shape, head, tail, dim, info, describe, ]
"""
# badCars = cars.iloc[10: 15]
# get the 3 car 
# print(badCars)
# print("shape", cars.shape)
# print("info", cars.info)
# print("dim & size", cars.size, cars.ndim)
# print("info", cars.info)
# print("dataframe describe", cars.describe)
# print("value_counts", cars.symboling.value_counts())
# print("columns", cars.columns)
# print('df statistics - describe()', cars.describe())
# print around 10 row of data from 3 cols
selected_cols = ['highway-mpg', 'horsepower', 'engine-size'] #this array of string were selected to practice working on accessing dataframe data
selectedRows = carsWithNumericalInfo.loc[:9, selected_cols] #select the first 10 rows with only the selected columns from data using loc
selectedRowsAllCols = carsWithNumericalInfo.loc[:9, :]
carsWithNumericalInfo = carsWithNumericalInfo.replace("?", np.nan) #replace is a method of the dataframe used to change the value that meets the condition into another value
first5NumericalWithNAN = carsWithNumericalInfo.head(5) 
carsWithNumericalInfo = carsWithNumericalInfo.astype('float')
carsWithNumericalInfo = carsWithNumericalInfo.dropna(subset=['price'])
carsWithNumericalInfo = carsWithNumericalInfo.fillna(carsWithNumericalInfo.mean())

# After data preprocessing, the carsWithNumericalData has data that doesn't contain any null values, and any not an number value is filled with mean for that column

# Normalizing all column to 0-1
priceCol = carsWithNumericalInfo['price']
carsWithNumericalInfo = (carsWithNumericalInfo - carsWithNumericalInfo.min()) / (carsWithNumericalInfo.max() - carsWithNumericalInfo.min())
carsWithNumericalInfo['price'] = priceCol

# Data Cleaning
def knn_train_test(train_col, target_col, df):
        np.random.seed(1) #setting the seed for np.random to ensure the sequence of generated random numbers are repeated each time the program is run
        
        shuffled_index = np.random.permutation(df.index) # this will shuffle the array in a way where df.index usually is an array of indices
        rand_df = df.reindex(shuffled_index) #shuffled_index is an numpy array of shuffled data that when passed in to reindex just reindex the original dataframe
        
        # Divide the total number of rows into half and round it up "why are we rounding, and dividing it?" (we want to split the data into training and testing)
        last_train_row = int(len(rand_df) / 2)
        
        train_df = rand_df.iloc[0:last_train_row]
        test_df = rand_df.iloc[last_train_row:]
        
        kValues = np.arange(1, 26) #use 1-25 but don't list them out
        kRMSES = {}
        
        for k in kValues:
                knn = KNeighborsRegressor(n_neighbors=k) #instantiate the KNN class 
                
                # train_df is access with double bracket as the 
                knn.fit(train_df[[train_col]], train_df[target_col]) #train_df with double [[]]
                
                predictedLabels = knn.predict(test_df[[train_col]])
                
                # Calculate and return RMSE
                mse = mean_squared_error(test_df[target_col], predictedLabels)
                rmse = np.sqrt(mse)
                kRMSES[k] = rmse
        return kRMSES

rmseResults = {}
carsWithFeaturesInfoOnly = carsWithNumericalInfo.loc[:, carsWithNumericalInfo.columns != 'price']

# for each feature which is a column in the dataset, get the Root mean square error of each k values from 1-25. Store that in rmseREsults with key feature.
# rmseResults will be an object with a key being feature and each key will have a value of an object.
for feature in carsWithFeaturesInfoOnly:
        rmse_val = knn_train_test(feature, 'price', carsWithNumericalInfo)
        rmseResults[feature] = rmse_val
        
featureAvgRMSE = {} 
# print("rmseResults", rmseResults)
for k,v in rmseResults.items(): #rmseResults contains the RMSE of each k value per feature
        avgRMSE = np.mean(list(v.values())) #the k are the feature, and the v are the RMSE of each k values for the feature. v is an object, v.values() gets all value into an array
        featureAvgRMSE[k] = avgRMSE #featureAvgRMSE will store the avgRMSE for each feature
        
seriesAvgRMSE = pd.Series(featureAvgRMSE) #turns everything into a series, before its just an object
sortedSeriesAvgRMSE = seriesAvgRMSE.sort_values() #sort it
sortedFeatures = sortedSeriesAvgRMSE.index

#train_col is now an array of features instead of 1
def knn_train_test(train_col, target_col, df):
        np.random.seed(1) #setting the seed for np.random to ensure the sequence of generated random numbers are repeated each time the program is run
        
        shuffled_index = np.random.permutation(df.index) # this will shuffle the array in a way where df.index usually is an array of indices
        rand_df = df.reindex(shuffled_index) #shuffled_index is an numpy array of shuffled data that when passed in to reindex just reindex the original stuff
        
        # Divide the total number of rows into half and round it up "why are we rounding, and dividing it?" (we want to split the data into training and testing)
        last_train_row = int(len(rand_df) / 2)
        
        train_df = rand_df.iloc[0:last_train_row]
        test_df = rand_df.iloc[last_train_row:]
        
        kValues = np.arange(1, 26) #use 1-25 but don't list them out
        kRMSES = {}

        for k in kValues:
                knn = KNeighborsRegressor(n_neighbors=k) #instantiate the KNN class 
                knn.fit(train_df[train_col], train_df[target_col]) #this line use the built in fit method to fit the training data that are basically a lis
                
                predictedLabels = knn.predict(test_df[train_col])
                
                # Calculate and return RMSE
                mse = mean_squared_error(test_df[target_col], predictedLabels)
                rmse = np.sqrt(mse)
                kRMSES[k] = rmse
        return kRMSES

k_rmse_results = {}

# for each number from 2-7. We will get that many feature to test our k_nearest_neighbor model. Which will return the rmse for all 26 k values from 1-25
for nr_best_feats in range(2,7):
    print(sortedFeatures[:nr_best_feats])
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sortedFeatures[:nr_best_feats],
        'price',
        carsWithNumericalInfo
    )

# we will then plot the valuees of all 26 k values of each number of eatures into the plot
for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())  
    plt.plot(x,y, label="{}".format(k))
    
plt.xlabel('k value')
plt.ylabel('RMSE')
plt.legend()       
plt.show() 
        
        
        
        
        