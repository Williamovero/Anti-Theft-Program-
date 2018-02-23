import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv('dataming2.csv')
a=pd.to_numeric(df['age'])
a.head()

# Counting frequency and percentage of frad
df_fraud= df[df['fraud']==1]
num_transaction_total, num_transaction_fraud = len(df), len(df_fraud)
num_transaction_total, num_transaction_fraud

percent_fraud = round(num_transaction_fraud / num_transaction_total * 100, 2)
percent_safe = 100 - percent_fraud
percent_fraud, percent_safe

df.describe()
df.columns
cdf=df[['step','age', 'gender', 'amount', 'fraud']]
cdf.head(10)


Y = df[['fraud']]
X = df[['amount']]

Nc = range(1, 10)

kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score

pl.plot(Nc,score)
pl.show()

pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

kmeans=KMeans(n_clusters=2)
kmeansoutput=kmeans.fit(Y)
kmeansoutput
pl.figure('2 Cluster K-Means')
pl.scatter(pca_d[:, 0], pca_c[:, 0], c=kmeansoutput.labels_)
pl.xlabel('fraud')
pl.ylabel('amount')
pl.title('2 Cluster K-Means')
pl.show()

Y = df[['fraud']]
X = df[['age']]

Nc = range(1, 4)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score

pl.plot(Nc,score)
pl.title('Clusters between age and fraud')
pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

kmeans=KMeans(n_clusters=2)
kmeansoutput=kmeans.fit(Y)
kmeansoutput
pl.figure('2 Cluster K-Means')
pl.scatter(pca_d[:, 0], pca_c[:, 0], c=kmeansoutput.labels_)
pl.xlabel('fraud')
pl.ylabel('age')
pl.title('2 Cluster K-Means')
pl.show()

Y = df[['amount']]
X = df[['age']]

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score

pl.plot(Nc,score)
pl.title('Clusters between age and amount')

pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

kmeans=KMeans(n_clusters=20)
kmeansoutput=kmeans.fit(Y)
kmeansoutput
pl.figure('2 Cluster K-Means')
pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
pl.xlabel('Age')
pl.ylabel('Amount')
pl.title('20 Cluster K-Means')
pl.show()

