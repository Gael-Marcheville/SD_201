#%% import libs
import pandas as pd 

#%% import raw data
df_churn = pd.read_csv("./dataset/employee_churn_data.csv") #raw data, 10 000 rows

#%% clean data

dfc_churn = df_churn.copy() #deep_copy of dataset
dfc_churn = dfc_churn.dropna() #Non-assigned values are handled

## create values
## create id_department
dfc_department = pd.DataFrame({'department' : df_churn['department']})
dfc_department.drop_duplicates(keep = 'first', inplace=True)
dfc_department['id_department'] = dfc_department.index
## create cat_salary
dfc_salary = pd.DataFrame({'salary' : df_churn['salary']})
dfc_salary.drop_duplicates(keep = 'first', inplace=True)
dfc_salary['cat_salary'] = [2 if s == 'high' else 1 if s == 'medium' else 0 for s in dfc_salary['salary']]  
## create is_left
dfc_left = pd.DataFrame({'left' : df_churn['left']})
dfc_left.drop_duplicates(keep = 'first', inplace=True)
dfc_left['is_left'] = [1 if s == 'yes' else 0 for s in dfc_left['left']]

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

## format
dfc_churn['tenure'] = [int(s) for s in dfc_churn['tenure']]  

#%% select training_data

#%% select test_data

display(dfc_churn)
dfc_churn.to_csv(r"./result.csv")