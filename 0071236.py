#Importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

#Importing data
data_set = pd.read_csv('./hw01_images.csv', header=None)
labels = pd.read_csv('./hw01_labels.csv', header=None)

#Adjusting labels for concatenating data_set and labels
labels = labels.rename(columns = {0: 4096})
concatted_df = pd.concat([data_set, labels], axis = 1)

#Train and test splitting
train = concatted_df[0:200]
test = concatted_df[200:400]

#Estimating mean, deviation parameters and prior probabilities to create a model
mean = pd.DataFrame([train[range(0, 4096)][train[4096] == 1].mean(), train[range(0, 4096)][train[4096] == 2].mean()])
std_dev = pd.DataFrame([train[range(0, 4096)][train[4096] == 1].std(), train[range(0, 4096)][train[4096] == 2].std()])
prior_prob = pd.DataFrame([train[4096][train[4096] == 1].size/train[4096].size, train[4096][train[4096] == 2].size/train[4096].size])

#Creating model for train dataset.
score_val_1 = -0.5 * np.log(2 * np.pi * std_dev.iloc[0,:]**2) - 0.5 * (train[range(0, 4096)] - mean.iloc[0,:])**2 / std_dev.iloc[0,:]**2

score_val_2 = -0.5 * np.log(2 * np.pi * std_dev.iloc[1,:]**2) - 0.5 * (train[range(0, 4096)] - mean.iloc[1,:])**2 / std_dev.iloc[1,:]**2

score_val = pd.concat([score_val_1.sum(axis=1), score_val_2.sum(axis=1)], axis=1)
score_val[0] = score_val[0].add(np.log(prior_prob.iloc[0,0]))
score_val[1] = score_val[1].add(np.log(prior_prob.iloc[1,0]))

#Printing results on confusion matrix for train data points.
print(confusion_matrix(train[4096], score_val.idxmax(axis=1)+1))

#Creating model for test dataset.

score_val_1 = -0.5 * np.log(2 * np.pi * std_dev.iloc[0,:]**2) - 0.5 * (test[range(0, 4096)] - mean.iloc[0,:])**2 / std_dev.iloc[0,:]**2

score_val_2 = -0.5 * np.log(2 * np.pi * std_dev.iloc[1,:]**2) - 0.5 * (test[range(0, 4096)] - mean.iloc[1,:])**2 / std_dev.iloc[1,:]**2

score_val = pd.concat([score_val_1.sum(axis=1), score_val_2.sum(axis=1)], axis=1)
score_val[0] = score_val[0].add(np.log(prior_prob.iloc[0,0]))
score_val[1] = score_val[1].add(np.log(prior_prob.iloc[1,0]))

#Printing results on confusion matrix for test data points.
print(confusion_matrix(train[4096], score_val.idxmax(axis=1)+1))
