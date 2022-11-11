#%% import
import pandas as pd 
import clean_churn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import pydot

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

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 25, min_samples_split = 10, max_depth = 25, min_samples_leaf = 6, bootstrap = True, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

#%% improve model

# param_grid = { 
#     'max_features': ['auto', 'sqrt', 'log2']
#     'bootstrap': [True, False],
#     'max_depth': [20, 25, 30, 35, None],
#     'min_samples_leaf': [4, 6, 8 ],
#     'min_samples_split': [ 10, 15, 20],
#     'n_estimators': [20,25,30,35]
#  }

#rf_CV = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
#rf_CV.fit(train_features, train_labels)
#print(rf_CV.best_params_)

#%% Make prediction

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

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

display(pd.DataFrame(rf.feature_importances_, index = feature_list, columns = ["importance"]).sort_values("importance", ascending = False))

#%% Print trees

# tree = rf.estimators_[3]# Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
# graph.write_png('./result/example_of_random_forest_tree.png')