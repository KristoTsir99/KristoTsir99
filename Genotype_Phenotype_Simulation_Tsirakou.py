# The code below aims to simulate genotype and phenotype data, conduct a heritability control,
# and visualize the respective SNPs.
# The code was adapted from ChatGPT.
#For further information, you may visit https://openai.com/chatgpt

###########################################################################

# First, we import the required packages/modules for our simulations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Then, we proceed with the simulation of the genotypes
# In this case, we will simulate genotypes for 400 individuals and 20 SNPs
individuals = 400
SNPs = 20

# The generated genotypes will be random
# Each SNP, depending on the number of minor alleles, will be assigned the number 0, 1, or 2 
Genotypes = np.random.randint(0, 3, size=(individuals, SNPs))

# Showing the genotypes (for example, let's assume we want to check the 20 first individuals)

print("Genotypes:\n", Genotypes[:20])

# Now, we will be defining and assigning genetic effects, namely beta values, 
# to each single-nucleotide polymorphism
Beta = np.random.normal(0, 1, size=SNPs)

# Showing the effect sizes
print("Effect sizes (Beta):", Beta)

# Now, we will proceed with the simulation of the phenotypes
# The phenotypes will be generated as a linear combination of the polymorphisms
# in addition to the noise
Noise = np.random.normal(0, 1, size=individuals)
Phenotypes = Genotypes @ Beta + Noise

# Showing the phenotypes (for example, let's assume we want to check the 20 first individuals)
print("Phenotypes:\n", Phenotypes[:20])

# Having the genotypes and phenotypes, now we will proceed  with the heritability control

# First, we store the heritability control into a variable
heritability_control =0.6

def simulate_phenotype_with_heritability(Genotypes, Beta, heritability_control=0.6):
    genetic = Genotypes @ Beta
    variance_genetic = np.var(genetic)
    variance_environmental = variance_genetic * (1 - heritability_control) / heritability_control
    noise = np.random.normal(0, np.sqrt(variance_environmental), size=Genotypes.shape[0])
    return genetic + noise

# Showing the output
Phenotypes_heritability_control = simulate_phenotype_with_heritability(Genotypes, Beta, heritability_control=0.6)

# Visualizing the above
plt.hist(Phenotypes_heritability_control, bins=20, color='blue', edgecolor='black')
plt.title("Phenotype Distribution (heritability_control = 0.6)")
plt.xlabel("Phenotype Value")
plt.ylabel("Frequency")
plt.show()