import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

n_cluster = 8

dfc_churn = clean_churn.get_dfc_churn()

df = dfc_churn.drop(["has_left"], axis=1)
kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0)

df_mean = kmeans.fit_predict(df)

index = 0
left = list(dfc_churn["has_left"])
prop_left = left.count(1) / len(left)

prop_mean = [0 for i in range(n_cluster)]
nb = [list(df_mean).count(i) for i in range(n_cluster)]
for i in df_mean:
    if left[index] == 1:
        prop_mean[i] += 1     
    index += 1
    
for i in range(n_cluster):
    prop_mean[i] /= list(df_mean).count(i)

infos_clusters = [(prop_mean[i], nb[i]) for i in range(n_cluster)]

print(infos_clusters)

#%%

### SELECTION OF DEPARTMENTS ###
dfc_department = pd.DataFrame({'department' : dfc_churn['id_department']})
dfc_department.drop_duplicates(keep = 'first', inplace=True)
departments = list(dfc_department.department)

n_cluster = 4

infos_departments = []
kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0)
kmeans_dic = {}
### TRAINING ###
for id_department in departments:
    dfc_filtered = dfc_churn[dfc_churn['id_department'] == id_department]
    dfc_filtered = dfc_filtered.drop(["id_department"], axis=1)
    left = list(dfc_filtered["has_left"])

    dfc_filtered = dfc_filtered.drop(["has_left"], axis=1)
    
    df_mean = kmeans.fit_predict(dfc_filtered)
    kmeans_dic[str(id_department)] = df_mean
    ### ANALYSIS OF PROPORTIONS ###
    index = 0
    prop_left = left.count(1) / len(left)
    print(prop_left)
    prop_mean = [0 for i in range(n_cluster)]
    nb = [list(df_mean).count(i) for i in range(n_cluster)]
    for i in df_mean:
        if left[index] == 1:
            prop_mean[i] += 1     
        index += 1
        
    for i in range(n_cluster):
        prop_mean[i] /= list(df_mean).count(i)
    
    infos_clusters = [(prop_mean[i], nb[i]/sum(nb)) for i in range(n_cluster)]
    infos_departments.append(infos_clusters)
    print(infos_clusters)    
    
#%%
### HISTOGRAMM PLOT OF CARACTERISTICS PER DEPARTMENT ###
index = 0
plt.figure(figsize =(20,15))

largeur = 0.3


for id_department in departments:
    clusters = np.array(range(n_cluster))
        
    left_proportion = [l[0] for l in infos_departments[index]]
    population_proportion = [l[1] for l in infos_departments[index]]
        
    ax = plt.subplot(3, 3, index +1)
    r1 = ax.bar(clusters - largeur/2, left_proportion, largeur)
    r2 = ax.bar(clusters + largeur/2, population_proportion, largeur)
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters)
        
    plt.title(f"department id : {id_department}")
    plt.legend(["Proportion of left people", "Population (normalized)"])
    
    index += 1

plt.show()

    
#%%

N = 11

wcss=[]
for i in range(1,N):
    print(i)
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
     
x_optim = 4
plt.plot(list(range(1,N)),wcss)
plt.axvline(x=x_optim, ymin=0, ymax=0.2,color='red',linestyle='--')

plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()