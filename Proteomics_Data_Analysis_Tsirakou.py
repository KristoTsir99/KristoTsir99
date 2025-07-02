# The code below aims to perform proteomics data analysis (with artificial data), 
# pre-process them, statistically analyze them, and visualize the results.  
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

# Loading the required packages/modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# Random seed for reproducibility
np.random.seed(50)

# Artifcial proteomics data

# Variables
number_proteins = 800
number_samples_per_group = 20

# Simulate data
control_group = np.random.normal(loc=20, scale=2, size=(number_proteins, number_samples_per_group))
treatment_group = np.random.normal(loc=20, scale=2, size=(number_proteins, number_samples_per_group))

# What about differential expression in 60 proteins?
differential_expression_indices = np.random.choice(number_proteins, 60, replace=False)
treatment_group[differential_expression_indices] += np.random.normal(3, 0.5, size=(60, number_samples_per_group))

# Combining the data
data_combined = np.concatenate([control_group, treatment_group], axis=1)
columns = [f'Control_Group_{i+1}' for i in range(number_samples_per_group)] + [f'Treatment_Group_{i+1}' for i in range(number_samples_per_group)]
data_frame_combined = pd.DataFrame(data_combined, index=[f'Proteins_{i+1}' for i in range(number_proteins)], columns=columns)

# Preprocessing the data - Normalization
data_frame_combined_log2 = np.log2(data_frame_combined + 1)

# Z-Score normalization
data_frame_combined_zscore = data_frame_combined_log2.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# Exploratory Data Analysis - Principal Component Analysis (PCA)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_frame_combined_log2.T)

principal_component = PCA(n_components=2)
components = principal_component.fit_transform(X_scaled)

# Data frame for plotting pursposes
principal_component_df = pd.DataFrame(components, columns=['Component 1', 'Component 2'])
principal_component_df['Groups'] = ['Control Group'] * number_samples_per_group + ['Treatment Group'] * number_samples_per_group

# Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(data=principal_component_df, x='Component 1', y='Component 2', hue='Groups', s=100)
plt.title("Principal Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()

# Create a heatmap of the 60 most variable proteins

top_variable_proteins = data_frame_combined_log2.var(axis=1).sort_values(ascending=False).head(60).index
plt.figure(figsize=(10, 8))
sns.heatmap(data_frame_combined_log2.loc[top_variable_proteins], cmap="viridis", xticklabels=True)
plt.title("Top 60 Most Variable Proteins")
plt.show()

# Performing differential expression analysis

# t-test for the association between the control and treatment group
pvals = []
log2fc = []

for protein in data_frame_combined_log2.index:
    control_group_vals = data_frame_combined_log2.loc[protein, data_frame_combined_log2.columns.str.contains("Control_Group")]
    treatment_group_vals = data_frame_combined_log2.loc[protein, data_frame_combined_log2.columns.str.contains("Treatment_Group")]
    
    stat, p = ttest_ind(control_group_vals, treatment_group_vals)
    pvals.append(p)
    log2fc.append(treatment_group_vals.mean() - control_group_vals.mean())

# Combining the above results
results = pd.DataFrame({
    'Protein': data_frame_combined_log2.index,
    'log2FC': log2fc,
    'p-value': pvals
}).set_index('Protein')

# Adding adjusted p-values (Benjamini-Hochberg)
results = results.sort_values('p-value')
results['adj-p'] = results['p-value'] * len(results) / (np.arange(1, len(results)+1))
results['adj-p'] = results['adj-p'].clip(upper=1)

# Visualization with a volcano plot

plt.figure(figsize=(10, 6))
sns.scatterplot(data=results, x='log2FC', y=-np.log10(results['p-value']), hue=results['adj-p'] < 0.05, palette={True: "yellow", False: "green"})
plt.title("Volcano Plot")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 p-value")
plt.axhline(-np.log10(0.05), color='blue', linestyle='--')
plt.axvline(1, color='green', linestyle='--')
plt.axvline(-1, color='green', linestyle='--')
plt.grid(True)
plt.show()








