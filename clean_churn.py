#%% import libs
import pandas as pd 

#%% import raw data
df_churn = pd.read_csv("./dataset/employee_churn_data.csv") #raw data, 10 000 rows

#%% clean data

dfc_churn = df_churn.copy() #deep_copy of dataset
dfc_churn = dfc_churn.dropna() #Non-assigned values are handled

## format
dfc_churn['tenure'] = [int(s) for s in dfc_churn['tenure']] #passage en int car valeurs entières => optimisation pour les calculs  

## delete outliers values
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.promoted != 0) & (dfc_churn.promoted != 1)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.review < 0) | (dfc_churn.review > 1)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.projects < 0) | (dfc_churn.projects > 100)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.salary != 'high') & (dfc_churn.salary != 'low') & (dfc_churn.salary != 'medium')].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.tenure < 0) | (dfc_churn.tenure > 100)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.satisfaction < 0) | (dfc_churn.satisfaction > 1)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.bonus != 0) & (dfc_churn.bonus != 1)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.avg_hrs_month < 0) | (dfc_churn.avg_hrs_month > 744)].index)
dfc_churn = dfc_churn.drop(dfc_churn[(dfc_churn.left != 'yes') & (dfc_churn.left != 'no')].index)
dfc_churn = dfc_churn.reset_index(drop=True)

## create values
## create id_department
dfc_department = pd.DataFrame({'department' : df_churn['department']})
dfc_department.drop_duplicates(keep = 'first', inplace=True)
dfc_department = dfc_department.reset_index(drop=True)
dfc_department['id_department'] = dfc_department.index
## create cat_salary
dfc_salary = pd.DataFrame({'salary' : df_churn['salary']})
dfc_salary.drop_duplicates(keep = 'first', inplace=True)
dfc_salary['cat_salary'] = [2 if s == 'high' else 1 if s == 'medium' else 0 for s in dfc_salary['salary']]  
## create is_left
dfc_left = pd.DataFrame({'left' : df_churn['left']})
dfc_left.drop_duplicates(keep = 'first', inplace=True)
dfc_left['has_left'] = [1 if s == 'yes' else 0 for s in dfc_left['left']]

## merge values
#id_department
dfc_churn = pd.merge(dfc_department, dfc_churn, how='inner', on=['department'])
dfc_churn = dfc_churn.drop(["department"],axis = 1)
#cat_salary
dfc_churn = pd.merge(dfc_salary, dfc_churn, how='inner', on=['salary'])
dfc_churn = dfc_churn.drop(["salary"],axis = 1)
#is_left
dfc_churn = pd.merge(dfc_left, dfc_churn, how='inner', on=['left'])
dfc_churn = dfc_churn.drop(["left"],axis = 1)

#%% select training_data

dfc_churn_train = dfc_churn.copy()
ids_test_list = [int(line.strip()) for line in open('ids_test.txt', 'r')]

for i in ids_test_list:
    dfc_churn_train = dfc_churn_train.drop(i)

#%% select test_data

dfc_churn_test = dfc_churn.copy()
ids_test_list = [int(line.strip()) for line in open('ids_test.txt', 'r')]
ids_training_list = [j for j in [i for i in range(len(dfc_churn))] if j not in ids_test_list]

for i in ids_training_list:
    dfc_churn_test = dfc_churn_test.drop(i)

#%% export in csv
#dfc_churn.to_csv(r"./result/dfc_churn.csv")

#%% function for import

def get_dfc_churn_train():
    return dfc_churn_train

def get_dfc_churn_test():
    return dfc_churn_test

def get_raw_df_churn():
    return df_churn

def get_dfc_churn():
    return dfc_churn