# The code below aims to plot synthetic archetypes with annotated blood metabolites.
# The main objective is to investigate the relationship between these 2 
# parameters (namely, archetypes and blood metabolites).

# The code was adapted from ChatGPT and the data used are generated with its help.
# For further information, you may visit https://openai.com/chatgpt

######################################################################

# First, we import the required packages/modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Generating synthetic data (i.e., archetypes, metabolites, etc.)
number_samples = 80
number_metabolites = 40
Archetypes = ['Archetype 1', 'Archetype 2', 'Archetype 3']
np.random.seed(0)

# Step 2: Synthetic metabolite values (with a normal distribution)
Metabolite_values = np.random.normal(loc=0, scale=1, size=(number_samples, number_metabolites))

# Step 3: Synthetic archetypes
Archetype_labels = np.random.choice(Archetypes, size=number_samples)

# Step 4: Annotating the metabolites
Metabolite_names = [f"Met_{i+1}" for i in range(number_metabolites)]
Annotations = [f"Pathway_{np.random.randint(1,5)}" for _ in range(number_metabolites)]  # e.g., metabolic pathway labels

# Step 5: Creating a dataframe for the synthetic data
data_frame = pd.DataFrame(Metabolite_values, columns=Metabolite_names)
data_frame['Archetype'] = Archetype_labels

# Step 6: Creating a dataframe for the metabolite annotation
Annotations_data_frame = pd.DataFrame({
    'Metabolite': Metabolite_names,
    'Annotation': Annotations
}).set_index('Metabolite')

# Step 7: Creating the group means

Group_means = data_frame.groupby('Archetype').mean()

# Step 8: Pre-processing (Z-score standardization) 
Scaler = StandardScaler()
Group_means_scaled = pd.DataFrame(
    Scaler.fit_transform(Group_means),
    index=Group_means.index,
    columns=Group_means.columns
)

# Creating a heatmap showing the relationship between the archetypes 
# and the annotated metabolites

# Step 10: Adding annotation to y-axis 
Metabolite_labels = [f"{met} ({Annotations_data_frame.loc[met, 'Annotation']})" for met in Group_means_scaled.columns]
Group_means_scaled.columns = Metabolite_labels

# Step 11: Plotting
plt.figure(figsize=(14, 8))
sns.heatmap(Group_means_scaled.T, 
            cmap='coolwarm', 
            center=0, 
            xticklabels=True, 
            yticklabels=True, 
            cbar_kws={"label": "Z-score"})

plt.title("Archetype vs.Annotated Blood Metabolites", fontsize=16)
plt.xlabel("Archetype")
plt.ylabel("Annotated Metabolites")
plt.tight_layout()
plt.show()
