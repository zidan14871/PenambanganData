import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dataset sample
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 55, 57, 58, 60, 65],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 94, 50, 60, 70]
}

df = pd.DataFrame(data)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Annual_Income', 'Spending_Score']])

print("=== K-Means Customer Segmentation ===")
print(df)

# Visualize
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering Result')
plt.show()
