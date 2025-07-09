# The code below aims to perform archetypal analysis on synthetic artificial datasets
# and help me understand how this type of analysis is performed and its significance in
# research. 
# I have practised this type of analysis with RStudio
# and now I will perfor it in Python. 
# As a proxy for archetypal analysis, I will perform NMF 
# (Non-negative MatrixFactorization).
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

##############################################################

# First, we import the required modules/packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF

# We set a random seed for reproducibility
np.random.seed(38)

# Generating synthetic data: 4 clusters in a 2-dimensional space
samples = 250
Centers = np.array([
    [0, 0],
    [5, 5],
    [0, 5],
    [5, 0]
])

X_list = []
for center in Centers:
    cluster = np.random.normal(loc=center, scale=0.7, size=(samples // 4, 2))
    X_list.append(cluster)

X = np.vstack(X_list)

# Then, we shift the data to be non-negative (NMF requires non-negative values)
X_min = X.min()
if X_min < 0:
    X_shifted = X - X_min
else:
    X_shifted = X

# Performing NMF as a proxy for archetypal analysis (if we wanted to perform
# archetypal analysis without NMF as a proxy, we would use different packages)
k = 2
nmf_model = NMF(n_components=k, init='random', random_state=38, max_iter=1000)
W = nmf_model.fit_transform(X_shifted)
H = nmf_model.components_

# We reshift the archetypes to the original scale
archetypes_points = H.T + X_min

# Then, we prepare dataframes for plotting
data_frame = pd.DataFrame(X, columns=["X1", "X2"])
archetypes_data_frame = pd.DataFrame(archetypes_points, columns=["X1", "X2"])

# Plotting and Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data_frame, x="X1", y="X2", color="grey", alpha=0.5)
sns.scatterplot(data=archetypes_data_frame, x="X1", y="X2", color="red", s=100)

plt.title("NMF as Approximate Archetypal Analysis")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
