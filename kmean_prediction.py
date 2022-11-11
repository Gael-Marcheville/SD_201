import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_cluster = 2

# Recovery of dataframes
dfc_train = clean_churn.get_dfc_churn_train()
dfc_train = dfc_train.drop(["has_left"], axis=1)

dfc_test = clean_churn.get_dfc_churn_test()
test_label = list(dfc_test["has_left"])
dfc_test = dfc_test.drop(["has_left"], axis=1)

kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0)

dfc_mean = kmeans.fit(dfc_train)

#%%
predictions = dfc_mean.predict(dfc_test)
print("people predicted to leave rate  : ", sum(predictions)/len(predictions))
print("people who left rate :", sum(test_label)/len(test_label))
error = abs(test_label - predictions)

print("error on prediction : ", sum(error) / len(error))