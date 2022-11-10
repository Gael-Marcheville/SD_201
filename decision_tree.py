#%% import
import clean_churn
import numpy as np
import pydot
from sklearn.tree import DecisionTreeClassifier, export_graphviz

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

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(train_features,train_labels)

#%% Make prediction

#Predict the response for test dataset
predictions = clf.predict(test_features)

#%% Evaluating model

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Pourcentage d erreur :', round(np.mean(errors), 2), '%')


errors = [max(predictions[i] - test_labels[i],0) for i in range (len(predictions))]
# Print out the mean absolute error (mae)
print('Error when predict "has left" ', round(np.sum(errors), 2),)

errors = [max(- predictions[i] + test_labels[i],0) for i in range (len(predictions))]
# Print out the mean absolute error (mae)
print('Error when predict "has not left" ', round(np.sum(errors), 2),)

#%% Print trees

export_graphviz(clf, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
graph.write_png('./result/example_decision_tree.png')