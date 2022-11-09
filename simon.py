import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_cluster = 30

dfc_churn = clean_churn.get_dfc_churn()

df = dfc_churn.drop(["has_left"], axis=1)
kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0)

df_mean = kmeans.fit_predict(df)

index = 0
left = list(dfc_churn["has_left"])
prop_left = left.count(1) / len(left)

prop_mean = [0 for i in range(n_cluster)]

for i in df_mean:
    if left[index] == 1:
        prop_mean[i] += 1     
    index += 1
    
for i in range(n_cluster):
    prop_mean[i] /= list(df_mean).count(i)

print(prop_mean)

#%%
# Ca marche pas de prendre tout. On regarde par département :





#%%

wcss=[]
for i in range(1,40):
    print(i)
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
     
plt.plot(list(range(1,11)),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()