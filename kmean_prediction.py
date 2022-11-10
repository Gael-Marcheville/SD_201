import clean_churn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_cluster = 8

dfc_churn = clean_churn.get_dfc_churn()

df = dfc_churn.drop(["is_left"], axis=1)
kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0)

df_mean = kmeans.fit_predict(df)


