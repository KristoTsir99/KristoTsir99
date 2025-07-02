# The code below aims to perform proteomics data analysis (with artificial data), 
# pre-process them, statistically analyze them, and visualize the results.  
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt


# First, we install the required packages and load the respective libraries
install.packages("ggplot2")
install.packages("pheatmap")
install.packages("dplyr")
install.packages("tidyr")
install.packages("stats")
install.packages("ggrepel")

library(ggplot2)
library(pheatmap)
library(dplyr)
library(tidyr)
library(stats)
library(ggrepel)

# Then, we set a random seed for reproducibility
set.seed(40)

# Parameters of our artificial/synthetic data
number_proteins <- 300
number_samples_per_group <- 10

# Simulating the control and treatment groups
control_group <- matrix(rnorm(number_proteins * number_samples_per_group, mean = 10, sd = 1),
                        nrow = number_proteins)

treatment_group <- matrix(rnorm(number_proteins * number_samples_per_group, mean = 10, sd = 1),
                          nrow = number_proteins)

# Let's say that 18 proteins are differentially expressed
differential_expressed_proteins <- sample(1:number_proteins, 18, replace = FALSE)
treatment_group[differential_expressed_proteins, ] <- treatment_group[differential_expressed_proteins, ] + matrix(rnorm(18 * number_samples_per_group, mean = 3, sd = 2,
                                                                          nrow = 18)

# Combining the data
data_combined <- cbind(control_group, treatment_group)
colnames(data_combined) <- c(paste0("Control_Group", 1:number_samples_per_group),
                             paste0("Treatment_Group", 1:number_samples_per_group))
rownames(data_combined) <- paste0("Protein_", 1:number_proteins)

# Pre-processing the data (log2 transformation)

data_log2 <- log2(data_combined + 1)

# Z-score normalization
data_z <- t(apply(data_log2, 1, scale))
rownames(data_z) <- rownames(data_log2)

# Principal Component Analysis

PCA_results <- prcomp(t(data_log2), scale. = TRUE)
PCA_data_frame <- data.frame(PCA_results$x[, 1:2])
PCA_data_frame$Group <- factor(c(rep("Control", number_samples_per_group), rep("Treatment", number_samples_per_group)))

# Plotting the Principal Components
ggplot(PCA_data_frame, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(size = 4) +
  theme_minimal() +
  labs(title = "Principal Components of Proteomic Data")

# Generating the heatmap of the 18 differentially expressed proteins

# What is the variance?
protein_variance <- apply(data_log2, 1, var)
top18_differential_expression <- names(sort(protein_variance, decreasing = TRUE))[1:18]

# Plotting the heatmap
pheatmap(data_log2[top18_differential_expression, ],
         scale = "row",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         main = "Top 18 Most Variable Proteins",
         show_rownames = FALSE)

# Testing for differential expression 

group_labels <- c(rep("Control", number_samples_per_group), rep("Treatment", number_samples_per_group))

log2fc <- numeric(number_proteins)
pvals <- numeric(number_proteins)

for (i in 1:number_proteins) {
  control_vals <- data_log2[i, group_labels == "Control"]
  treatment_vals <- data_log2[i, group_labels == "Treatment"]
  
  t_test <- t.test(treatment_vals, control_vals)
  pvals[i] <- t_test$p.value
  log2fc[i] <- mean(treatment_vals) - mean(control_vals)
}

# Creating a data frame for the results
results <- data.frame(
  Protein = rownames(data_log2),
  log2FC = log2fc,
  p_value = pvals
)

# Adjusting the p-values (Benjamini-Hochberg)
results$adj_p <- p.adjust(results$p_value, method = "BH")

# Generating a volcano plot

results$Significant <- results$adj_p < 0.05

ggplot(results, aes(x = log2FC, y = -log10(p_value), color = Significant)) +
  geom_point() +
  scale_color_manual(values = c("green", "yellow")) +
  theme_minimal() +
  labs(title = "Volcano Plot", x = "Log2 Fold Change", y = "-Log10 p-value") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "darkgreen") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue")