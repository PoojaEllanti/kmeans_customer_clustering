import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.metrics import silhouette_score # type: ignore

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Rename columns for ease
df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spending'}, inplace=True)

# Select features
X = df[['Income', 'Spending']]

# Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method to find optimal K')
plt.grid(True)
plt.show()

# Fit KMeans with K=5 (based on elbow curve)
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Silhouette Score
score = silhouette_score(X, df['Cluster'])
print("Silhouette Score:", score)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Income', y='Spending', hue='Cluster', palette='Set2', s=100)
plt.title("Customer Segments by K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.legend()
plt.show()
