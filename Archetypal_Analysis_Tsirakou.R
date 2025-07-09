# The code below aims to perform archetypal analysis on synthetic artificial datasets
# and help me understand how this type of analysis is performed and its significance in
# research. 
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

##############################################################

# First, we install thee required packages (for the archetypal analysis anmd visualization)
if (!requireNamespace("archetypes", quietly = TRUE)) install.packages("archetypes")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")

# Then, we load the respective libraries
library(archetypes)
library(ggplot2)

# We then set a random seed for reproducibility
set.seed(38)

# We generate synthetic data.
# In this case, let's create 4 clusters in a 2-dimensional space
n <- 250
Centers <- matrix(c(0, 0, 5, 5, 0, 5, 5, 0), ncol = 2, byrow = TRUE)
X <- do.call(rbind, lapply(1:nrow(Centers), function(i) {
  cbind(rnorm(n / 4, Centers[i, 1], 0.7), rnorm(n / 4, Centers[i, 2], 0.7))
}))

# We continue by performing archetypal analysis with 2 archetypes
k <- 2
archetypal_analysis <- archetypes(X, k = k)

# We then extract the archetypes from the model
archetypes_points <- parameters(archetypal_analysis)
colnames(archetypes_points) <- c("X1", "X2")

# We then prepare our data for plotting purposes
df <- data.frame(X1 = X[, 1], X2 = X[, 2])
archetypes_df <- as.data.frame(archetypes_points)

# We plot the data points and the archetypes
ggplot(df, aes(x = X1, y = X2)) +
  geom_point(alpha = 0.5, color = "grey50") +
  geom_point(data = archetypes_df, aes(x = X1, y = X2), color = "red", size = 4) +
  ggtitle("Archetypal Analysis") +
  theme_minimal()
