import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


dfc_churn = clean_churn.get_dfc_churn()
dfc_churnn=dfc_churn.drop(["is_left"],axis = 1)


# set random seed for reproducibility
seed = 0

# remove stock names from the data
#X = data.drop(['StockName'], axis=1)

# train the kmeans algorithm on the data
kmeans_default = KMeans(n_clusters=2, random_state=seed).fit(dfc_churnn)

# calculate kmeans sse
sse_default = kmeans_default.inertia_

# print results
print('K-Means with default parameters:')
print('Sum of Squared Errors: {:.2f}'.format(sse_default))
y_kmeans = kmeans_default.fit_predict(dfc_churnn)

# k1=0
# k2=0
# for i in y_kmeans:
#     if i==0:
#         k1=k1+1
#     else:
#         k2=k2+1

# stayed=0
# for k in dfc_churn['is_left']:
#     if k==0:
#         stayed=stayed+1

# erreur=0
# g=0
# if y_kmeans[g]== dfc_churn[g]:
#     erreur=erreur+1
#     g=g+1


wcss=[]
#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and #appended to our wcss list.
for i in range(1,11): 
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
     
#i above is between 1-10 numbers. init parameter is the random #initialization method  
#we select kmeans++ method. max_iter parameter the maximum number of iterations there can be to 
#find the final clusters when the K-meands algorithm is running. we #enter the default value of 300
#the next parameter is n_init which is the number of times the #K_means algorithm will be run with
#different initial centroid.
kmeans.fit(dfc_churnn)
#kmeans algorithm fits to the X dataset
wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()    
    
    
    

