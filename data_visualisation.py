import clean_churn
import matplotlib.pyplot as plt
import numpy as np

dfc = clean_churn.get_dfc_churn()

#%% DEPARTMENTS

departments = dfc["id_department"]
departments_list = list(set(departments))

### CALCULATION OF PROPORTION PER DEPARTMENTS
prop_left_departments = []
labels = []
for id_department in departments_list:
     dfc_department = dfc[dfc['id_department'] == id_department]
     left = list(dfc_department["has_left"])
     prop_left_departments.append(sum(left))
     labels.append(str(100*sum(left)/len(left))[:5] + " %")

pop_departments = [list(departments).count(i) for i in departments_list]

plt.bar(x=departments_list, height=pop_departments)
plt.bar(x=departments_list, height=prop_left_departments, color = "red")
for x, y, l in zip(np.array(departments_list)-0.4, pop_departments, labels):
   plt.text(x, y, l)

plt.legend(["Population per departments", "Number of departures"])
plt.show()

#%% PROJECTS
projects = dfc["projects"]
project_list = list(set(projects))
plt.rcParams["figure.figsize"] = [5, 3.50]
prop_project = []
labels = []
for p in project_list:
     dfc_projects = dfc[dfc['projects'] == p]
     left = list(dfc_projects["has_left"])
     prop_project.append(sum(left))
     labels.append(str(100*sum(left)/len(left))[:5] + " %")
     
pop_projects = [list(projects).count(i) for i in project_list]

plt.bar(x=project_list, height=pop_projects)
plt.bar(x=project_list, height=prop_project, color = "red")

for x, y, l in zip(np.array(project_list)-0.2, pop_projects, labels):
   plt.text(x, y, l)

plt.legend(["Pop. of projects number", "Number of departures"])
plt.show()     

#%% SALARY
salary = dfc["cat_salary"]
salary_list = list(set(salary))
plt.rcParams["figure.figsize"] = [5, 3.50]
prop_salary = []
labels = []
for p in salary_list:
     dfc_salaries = dfc[dfc['cat_salary'] == p]
     left = list(dfc_salaries["has_left"])
     prop_salary.append(sum(left))
     labels.append(str(100*sum(left)/len(left))[:5] + " %")
     
pop_salaries = [list(salary).count(i) for i in salary_list]
plt.bar(x=salary_list, height=pop_salaries)
plt.bar(x=salary_list, height=prop_salary, color = "red")

for x, y, l in zip(np.array(salary_list)-0.2, pop_salaries, labels):
   plt.text(x, y, l)

plt.legend(["Pop. of salaries", "Departures"])
plt.show()  

#%%

salary = dfc["avg_hrs_month"]
salary_list = list(set(salary))

values, base = np.histogram(salary_list, bins=40)
cumulative = np.cumsum(values)
plt.plot(base[:-1], cumulative)

plt.xlabel("average hours per months")
plt.ylabel("Number of people")

