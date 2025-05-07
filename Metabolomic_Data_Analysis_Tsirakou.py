# The code below aims to simulate metabolomic data and analyze them
# by performing principal component analysis, k-means clustering for group sampling, and
# visualizing the results.  

# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

######################################################################

# First, we import the required packages/modules for our simulations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Then, we generate an artificial dataset which will contain our samples and
# metabolites
np.random.seed(40)

# Samples
samples = 200  
metabolites = 20

# For the metabolite concentrations (which may range from 10 to 100)
sample_data = np.random.rand(samples, metabolites) * 100

# We can create a data frame (to better structure our sample data)
metabolites = [f'metabolite_{i+1}' for i in range(metabolites)]
samples = [f'sample_{i+1}' for i in range(samples)]
sample_data_frame = pd.DataFrame(sample_data, columns=metabolites, index=samples)

# We perform Principal Component Analysis (PCA) to reduce the dimensionality
# of the data
PCA = PCA(n_components=2)
PCA_output = PCA.fit_transform(sample_data_frame)

# Visualizing the output of the PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=PCA_output[:, 0], y=PCA_output[:, 1], palette="viridis", s=200)
plt.title('PCA of the Metabolomic Dataset')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# K-means clustering for group sampling
K_Means = KMeans(n_clusters=4, random_state=40)
sample_data_frame['Cluster'] = K_Means.fit_predict(sample_data_frame)

# Visualizing the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=PCA_output[:, 0], y=PCA_output[:, 1], hue=sample_data_frame['Cluster'], palette='deep', s=200)
plt.title('K-means PCA')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Summary statistics
summary_statistics = sample_data_frame.describe()
print(summary_statistics)
