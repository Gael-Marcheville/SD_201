import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random


dfc_churn = clean_churn.get_dfc_churn()
dfc_churnn=dfc_churn.drop(["has_left"],axis = 1)
df_test=clean_churn.get_dfc_churn()


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

#%% new test sample
# mylist=[i for i in range(len(dfc_churn))]    
# random.shuffle(mylist)
# shuffle_list = mylist[:int(0.1*len(mylist))]

# with open('ids_test.txt', 'w') as temp_file:
#     for item in shuffle_list:
#         temp_file.write("%s\n" % item)

    
    

