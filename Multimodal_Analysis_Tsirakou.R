# The code below aims to simulate fMRI and metabolomic data and analyze them
# by performing pre-processing and visualization. The main objective is to
# extract meaningful information about the brain connectivity patterns and 
# the molecular profiling of various diseases,
# let's say Alzheimer's or Schizophrenia.  

# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt


# First, we install the required packages

install.packages("ggplot2")
install.packages("reshape")
install.packages("caret")
install.packages("randomForest")
install.packages("CCA")
install.packages("igraph")

# We then load the respective libraries

library(ggplot2)
library(reshape2)
library(caret)
library(randomForest)
library(CCA)
library(igraph)

# Step 1: We simulate some simulated fMRI and metabolomic data
data <- function(number_subjects = 300, number_regions = 50, number_metabolites = 20, disease_ratio = 0.5) {
set.seed(55) # we set a random seed for reproducibility
labels <- sample(c(0, 1), size = number_subjects, replace = TRUE, prob = c(1 - disease_ratio, disease_ratio))
  
upper_indices <- which(upper.tri(matrix(0, number_regions, number_regions)))
number_features <- length(upper_indices)
  
Fmri_data <- matrix(0, nrow = number_subjects, ncol = number_features)
Metabol_data <- matrix(0, nrow = number_subjects, ncol = number_metabolites)
  
  for (i in 1:number_subjects) {
    label <- labels[i]
    
    base_con <- matrix(runif(number_regions^2), nrow = number_regions)
    con_matrix <- (base_con + t(base_con)) / 2
    if (label == 1) {
      con_matrix <- con_matrix + diag(0.2, number_regions)
    } else {
      con_matrix <- con_matrix + diag(0.1, number_regions)
    }
    Fmri_data[i, ] <- con_matrix[upper.tri(con_matrix)]
    
    base_metabol <- rnorm(number_metabolites)
    if (label == 1) {
      base_metabol[1:5] <- base_metabol[1:5] + 2
    }
    Metabol_data[i, ] <- base_metabol
  }
  
  list(Fmri_data = Fmri_data, Metabol_data = Metabol_data, labels = labels)
}

# Step 2: Pre-processing of the simulated data
data_preprocessed <- function(Fmri_data, Metabol_data) {
Fmri_scaled <- scale(Fmri_data)
Metabol_scaled <- scale(Metabol_data)
list(Fmri_scaled = Fmri_scaled, Metabol_scaled = Metabol_scaled)
}

# Step 3: Dimensionality reduction by performing PCA
dimensions_reduced <- function(Fmri_scaled, Metabol_scaled, number_components = 10) {
Fmri_pca <- prcomp(Fmri_scaled)
Metabol_pca <- prcomp(Metabol_scaled)
list(Fmri_PCA = Fmri_pca$x[, 1:number_components], Metabol_PCA = Metabol_pca$x[, 1:number_components])
}

# Step 4: Canonical Correlation Analysis (CCA)
cca <- function(Fmri_PCA, Metabol_PCA, number_components = 10) {
cca_result <- cc(Fmri_PCA, Metabol_PCA)
cca_Fmri <- cca_result$scores$xscores[, 1:number_components]
cca_Metabol <- cca_result$scores$yscores[, 1:number_components]
list(cca_Fmri = cca_Fmri, cca_Metabol = cca_Metabol)
}

# Step 5: Performing classification using Random Forest
classify_multimodal_data <- function(cca_Fmri, cca_Metabol, labels) {
data_combined <- as.data.frame(cbind(cca_Fmri, cca_Metabol))
data_combined$label <- as.factor(labels)
  
set.seed(55)
train_index <- createDataPartition(data_combined$label, p = 0.8, list = FALSE)
train_data <- data_combined[train_index, ]
test_data <- data_combined[-train_index, ]
  
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100)
preds <- predict(rf_model, test_data)
  
cat("Confusion Matrix:\n")
print(confusionMatrix(preds, test_data$label))
}

# Step 6: Scatter plots for Canonical Correlation Analysis
plot_cca_scatter <- function(cca_Fmri, cca_Metabol, labels) {
data_frame_1 <- data.frame(CC1 = cca_Fmri[, 1], CC2 = cca_Fmri[, 2], Label = as.factor(labels))
data_frame_2 <- data.frame(CC1 = cca_Metabol[, 1], CC2 = cca_Metabol[, 2], Label = as.factor(labels))
  
plot_1 <- ggplot(data_frame_1, aes(x = CC1, y = CC2, color = Label)) + 
    geom_point() + 
    ggtitle("CCA - fMRI")
  
plot_2 <- ggplot(data_frame_2, aes(x = CC1, y = CC2, color = Label)) + 
    geom_point() + 
    ggtitle("CCA - Metabolomics")
  
  gridExtra::grid.arrange(plot_1, plot_2, ncol = 2)
}

# 7. Plot connectivity graph
plot_connectivity_graph <- function(avg_vector, n_regions = 50, title_str = "Connectivity") {
con_matrix <- matrix(0, nrow = n_regions, ncol = n_regions)
con_matrix[upper.tri(con_matrix)] <- avg_vector
con_matrix <- con_matrix + t(con_matrix)
  
g <- graph_from_adjacency_matrix(con_matrix, mode = "undirected", weighted = TRUE, diag = FALSE)
plot(g, layout = layout_with_fr(g), vertex.size = 5, edge.color = "grey", main = title_str)
}

# Running the whole pipeline

# Step 1: Simulating the data
dataset <- data()
Fmri_data <- dataset$Fmri_data
Metabol_data <- dataset$Metabol_data
labels <- dataset$labels

# Step 2: Pre-processing
scaled <- data_preprocessed(Fmri_data, Metabol_data)

# Step 3: Principal Component Analysis (PCA)
pca_results <- dimensions_reduced(scaled$Fmri_scaled, scaled$Metabol_scaled)

# Step 4: CCA
cca_results <- cca(pca_results$Fmri_PCA, pca_results$Metabol_PCA)

# Step 5: Random Forest Classification
classify_multimodal_data(cca_results$cca_Fmri, cca_results$cca_Metabol, labels)

# Step 6: CCA scatter plots
plot_cca_scatter(cca_results$cca_Fmri, cca_results$cca_Metabol, labels)

# Step 7: Connectivity graphs
average_healthy <- colMeans(Fmri_data[labels == 0, ])
average_diseased <- colMeans(Fmri_data[labels == 1, ])
plot_connectivity_graph(average_healthy, title_str = "Connectivity - Healthy")
plot_connectivity_graph(average_diseased, title_str = "Connectivity - Diseased")
