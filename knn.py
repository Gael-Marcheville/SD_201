#%% import

import clean_churn
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

#%% setup data

#full
# train
dfc_churn = clean_churn.get_dfc_churn()
# Labels are the values we want to predict
labels = np.array(dfc_churn['has_left'])# Remove the labels from the features
# axis 1 refers to the columns
features = dfc_churn.drop('has_left', axis = 1)# Saving feature names for later use
feature_list = list(features.columns)# Convert to numpy array
features = np.array(features)

# train
dfc_churn_train = clean_churn.get_dfc_churn_train()
# Labels are the values we want to predict
train_labels = np.array(dfc_churn_train['has_left'])# Remove the labels from the features
# axis 1 refers to the columns
train_features = dfc_churn_train.drop('has_left', axis = 1)# Saving feature names for later use
train_features = np.array(train_features)

## test
dfc_churn_test = clean_churn.get_dfc_churn_test()
# Labels are the values we want to predict
test_labels = np.array(dfc_churn_test['has_left'])# Remove the labels from the features
# axis 1 refers to the columns
test_features = dfc_churn_test.drop('has_left', axis = 1)# Saving feature names for later use
feature_list = list(test_features.columns)# Convert to numpy array
test_features = np.array(test_features)


#%% Learning

knn_model = KNeighborsClassifier(n_neighbors=25)
knn_model.fit(train_features, train_labels)

#%% Make prediction

predictions = knn_model.predict(test_features)

#%% Improve model

#Find best K for KNN method

error_rate = []

for i in range(0, 100):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(train_features, train_labels)
    predictions = knn_model.predict(test_features)
    errors = abs(predictions - test_labels)
    error_rate.append(np.mean(errors)*100)
    
plt.plot(range(0, 100),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
#Suite à plusieurs tests, on trouve que la meilleure valeur pour k est 40

#%%Bagging
best_knn = KNeighborsClassifier(n_neighbors=25)
bagging_model = BaggingClassifier(best_knn, n_estimators=100)
bagging_model.fit(train_features, train_labels)

predictions = bagging_model.predict(test_features)


#%% Evaluating model

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Pourcentage d erreur :', round(np.mean(errors)*100, 2), '%')


errors = [max(predictions[i] - test_labels[i],0) for i in range (len(predictions))]
# Print out the mean absolute error (mae)
print('Error when predict "has left" ', round(np.sum(errors), 2),)

errors = [max(- predictions[i] + test_labels[i],0) for i in range (len(predictions))]
# Print out the mean absolute error (mae)
print('Error when predict "has not left" ', round(np.sum(errors), 2),)

#%% confusion matrix

# Get and reshape confusion matrix data

# Visualise classical Confusion M0atrix
CM = confusion_matrix(test_labels, predictions)

# Visualize it as a heatmap
class_names = ['not left','left']

sn.heatmap(CM, annot=True, fmt=".0f", cmap="flare", annot_kws={"size":15})
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, fontsize = 15)
plt.yticks(tick_marks, class_names, rotation=90, fontsize = 15)
plt.title('Confusion Matrix for K-Nearest Neighbors with bagging', fontsize = 15)
plt.xlabel('Predicted label', fontsize = 15)
plt.ylabel('True label', fontsize = 15, rotation = 90)
plt.show()
