import pandas as pd 
import datetime as dt
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt

### import data
df_churn = pd.read_csv("./dataset/employee_churn_data.csv")

### clean data

dfc_churn = df_churn.copy()

## create id_department
dfc_department = pd.DataFrame({'department' : df_churn['department']})
dfc_department.drop_duplicates(keep = 'first', inplace=True)
dfc_department['id_department'] = dfc_department.index

## create id_salary
dfc_salary = pd.DataFrame({'salary' : df_churn['salary']})
dfc_salary.drop_duplicates(keep = 'first', inplace=True)
dfc_salary['id_salary'] = dfc_salary.index

## merge id_department
dfc_churn = pd.merge(dfc_department, dfc_churn, how='inner', on=['department'])
dfc_churn = dfc_churn.drop(["department"],axis = 1)

## merge id_salary
dfc_churn = pd.merge(dfc_salary, dfc_churn, how='inner', on=['salary'])
dfc_churn = dfc_churn.drop(["salary"],axis = 1)

## format
dfc_churn['left'] = [1 if s == 'yes' else 0 for s in dfc_churn['left']]  
dfc_churn['tenure'] = [int(s) for s in dfc_churn['left']]  

### select training_data

### select test_data

display(dfc_churn)
dfc_churn.to_csv(r"./result.csv")