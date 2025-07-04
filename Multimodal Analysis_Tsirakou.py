# The code below aims to simulate fMRI and metabolomic data and analyze them
# by performing pre-processing and visualization. The main objective is to
# extract meaningful information about the brain connectivity patterns and 
# the molecular profiling of various diseases,
# let's say Alzheimer's or Schizophrenia.  

# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

######################################################################

# First, we import the required modules/packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import networkx as nx

# The, we generate/simulate some fMRI and metabolomic data
def generate_data(number_subjects=200, number_regions=40, number_metabolites=15, disease_ratio=0.4):
    np.random.seed(35)

    labels = np.random.choice([0, 1], size=number_subjects, p=[1-disease_ratio, disease_ratio])

    Fmri_data = []
    Metabolomic_data = []

    for label in labels:
        # We create an fMRI connectivity matrix
        base_con = np.random.rand(number_regions, number_regions)
        con_matrix = (base_con + base_con.T) / 2

        # For the group speciric signal:
        if label == 1:  # state of disease
            con_matrix += np.eye(number_regions) * 0.2
        else:  # for the healthy state
            con_matrix += np.eye(number_regions) * 0.1

        Fmri_data.append(con_matrix[np.triu_indices(number_regions, k=1)])

# For the metabolomic data
        base_metabol = np.random.normal(0, 1, number_metabolites)
        if label == 1:
            base_metabol[:5] += 2  # let's upregulate some key metabolites
        Metabolomic_data.append(base_metabol)

    return np.array(Fmri_data), np.array(Metabolomic_data), labels

# Pre-processing the fMRI and metabolomic data:
def preprocess_data(Fmri_data, Metabolomic_data):
    scaler_Fmri = StandardScaler()
    scaler_Metabol = StandardScaler()
    Fmri_scaled = scaler_Fmri.fit_transform(Fmri_data)
    Metabol_scaled = scaler_Metabol.fit_transform(Metabolomic_data)
    return Fmri_scaled, Metabol_scaled

# Reducing the dimensionality of the fMRI and metabolomic data 
# (by performing principal component analysis - PCA)
def reduce_dimensions(Fmri_scaled, Metabol_scaled, number_components=8):
    PCA_Fmri = PCA(n_components=number_components)
    PCA_Metabol = PCA(n_components=number_components)
    Fmri_PCA = PCA_Fmri.fit_transform(Fmri_scaled)
    Metabol_PCA = PCA_Metabol.fit_transform(Metabol_scaled)
    return Fmri_PCA, Metabol_PCA, PCA_Fmri, PCA_Metabol

# Then, we perform canonical correlation analysis (CCA)
def run_cca(Fmri_PCA, Metabol_PCA, number_components=8):
    cca = CCA(n_components=number_components)
    cca_Fmri, cca_Metabol = cca.fit_transform(Fmri_PCA, Metabol_PCA)
    return cca_Fmri, cca_Metabol, cca

# Classifying fMRI and metabolomic data
def classify_multimodal_data(cca_Fmri, cca_Metabol, labels):
    Combined = np.hstack((cca_Fmri, cca_Metabol))
    X_train, X_test, y_train, y_test = train_test_split(Combined, labels, test_size=0.2, stratify=labels)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))

# Visualizing the results
def plot_cca_scatter(cca_Fmri, cca_Metabol, labels):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.scatterplot(x=cca_Fmri[:, 0], y=cca_Fmri[:, 1], hue=labels, palette='coolwarm')
    plt.title("Canonical Correlation Analysis - fMRI")
    plt.subplot(1,2,2)
    sns.scatterplot(x=cca_Metabol[:, 0], y=cca_Metabol[:, 1], hue=labels, palette='coolwarm')
    plt.title("Canonical Correlation Analysis - Metabolites")
    plt.tight_layout()
    plt.show()
    
# Creating a plot connectivity graph

def plot_connectivity_graph(average_con_vector, number_regions=40, title='Average Brain Connectivity'):
    con_matrix = np.zeros((number_regions, number_regions))
    iu = np.triu_indices(number_regions, k=1)
    con_matrix[iu] = average_con_vector
    con_matrix += con_matrix.T

    G = nx.from_numpy_array(con_matrix)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=35)
    nx.draw(G, pos, node_size=50, edge_color='black', alpha=0.5)
    plt.title(title)
    plt.show()

# Running the whole pipeline
Fmri_data, Metabolomic_data, labels = generate_data()

Fmri_scaled, Metabol_scaled = preprocess_data(Fmri_data, Metabolomic_data)
Fmri_PCA, Metabol_PCA, _, _ = reduce_dimensions(Fmri_scaled, Metabol_scaled)
cca_Fmri, cca_Metabol, _ = run_cca(Fmri_PCA, Metabol_PCA)
classify_multimodal_data(cca_Fmri, cca_Metabol, labels)
plot_cca_scatter(cca_Fmri, cca_Metabol, labels)

# Visualizing the average connectivity between each group
healthy_average = Fmri_data[labels == 0].mean(axis=0)
diseased_average = Fmri_data[labels == 1].mean(axis=0)
plot_connectivity_graph(healthy_average, title="Connectivity in Health")
plot_connectivity_graph(diseased_average, title="Connectivity in Disease")