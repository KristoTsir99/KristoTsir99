# The code below aims to identify variants of a specific gene (let's say
# the GLP1R gene) that influences glucose response.
# Archetypal analysis will be performed (with NMF as a proxy) and we
# will visualize how archetype wights are related to the phenotype,
# as well as the reconstruction error heatmap. 
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

##############################################################

# First off, we import the required modules/packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF, PCA # archetypal analysis and dimension reduction

# First step:: Simulating synthetic SNP data (related to the gene affecting glucose response)
np.random.seed(30) # random seed for reproducibility
number_samples = 200
number_snps = 20
snp_data = np.random.choice([0, 1, 2], size=(number_samples, number_snps))

# Simulating glucose response
coefficients = np.random.randn(number_snps)
noise = np.random.normal(0, 1, number_samples)
glucose_response = snp_data @ coefficients + noise

# Creating adataframe
df = pd.DataFrame(snp_data, columns=[f"SNP_{i+1}" for i in range(number_snps)])
df["glucose_response"] = glucose_response

# Second step: We use the MinMax scale (for NMF as a proxy for archetypal analysis)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["glucose_response"]))

# Third step: Principal Component Analysis (PCA) for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fourth step: Non-negative Matrix Factorization (NMF) for archetypal analysis
nmf = NMF(n_components=4, init='random', random_state=30, max_iter=1000)
W = nmf.fit_transform(X_scaled)
H = nmf.components_

# Fifth step: Computing ΔX (the reconstruction error while performing NMF through the iterations)
X_reconstructed = W @ H
Delta_X = X_scaled - X_reconstructed

# Sixth step: Ploting the reconstruction error heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(Delta_X, cmap="coolwarm", center=0, cbar_kws={'label': 'Reconstruction Error (ΔX)'})
plt.title("Reconstruction Error Heatmap (ΔX = Original - Reconstructed)")
plt.xlabel("SNP Features")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

# Seventh step: Merging our archetypal weights with the original data
df_archetypes = pd.DataFrame(W, columns=[f"Archetype_{i+1}" for i in range(W.shape[1])])
df_combined = pd.concat([df, df_archetypes], axis=1)

# Eighth step: Visualizing the archetypal weights with the glucose response
plt.figure(figsize=(8, 6))
for i in range(W.shape[1]):
    plt.scatter(df_combined[f"Archetype_{i+1}"], df_combined["glucose_response"], alpha=0.7, label=f'Archetype {i+1}')
plt.xlabel("Archetype Weight")
plt.ylabel("Glucose Response")
plt.title("Archetype Weights vs Glucose Response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()