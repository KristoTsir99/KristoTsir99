# The code below aims to apply Bayes' Theorem to real-world (or even imaginary) situations.
# First example: Let's assume we have a series of texts to read for an exam in Ancient Greek Philosophy but we are not sure 
# if all the texts we have are relevant with the topic.
# We know for sure that approximately 80% of the papers will be relevant, but there is a remaining 20% that is irrelevant. 
# The relevant papers will most probably contain the word 'Aristotle' compared to the irrelevant ones that might 
# mention it occasionally.
# Our aim is to update the probability that a text is philosophical, provided it contains the word 'Aristotle'.
# The prior probabilities are the following: the text to be or not to be philosophical.
# The likelihoods are the following: the text to contain the word 'Aristotle' provided it is philosophical (approximately 40%) and the text to contain the word 'Aristotle'
# provided it is not philosophical (approximately 15%).

#############################################################################

# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

#############################################################################


# Prior probabilities
text_philosophical <- 0.80  # the text is philosophical with a probability of 80% 
text_not_philosophical <- 0.20  # the text is not philosophical with a probability of 20%

# Likelihoods 
text_Aristotle_philosophical <- 0.40  # the text contains the word 'Aristotle' provided that is is philosophical in a probability of 40%
text_Aristotle_not_philosophical <- 0.15  # the text contains the word 'Aristotle' provided that it is not philosophical in a probability of 15%

# Evidence (or marginal likelihood)
text_Aristotle <- text_Aristotle_philosophical * text_philosophical + text_Aristotle_not_philosophical * text_not_philosophical # the probability of the text to contain the word 'Aristotle'

# Posterior probability
text_philosophical_Aristotle <- (text_Aristotle_philosophical * text_philosophical) / text_Aristotle  # the probability of the text to be philosophical provided it contains the word 'Aristotle'


# The output is stored into a variable (output)
output <- paste("Probability that the text is philosophical provided it contains the word 'Aristotle' is:", cat("Probability that the text is philosophical provided it contains the word 'Aristotle' is", round(text_philosophical_Aristotle, 4), "\n") )

############################################################################

# Now, we want to write a code where we provide a series of philosophical texts (in a csv file) and
# we aim to classify them as philosophical or not philosophical.

# First we install the required packages
install.packages("NLP") # natural language processing package
install.packages("tm")  # text mining package
install.packages("e1071")  # Naive Bayes classification package
install.packages("text2vec") # package that converts a text into a vector
install.packages("dplyr")  # package that, among others, provides tidy data frames (for viewing purposes)

# Then, we load the respective libraries
library(NLP)
library(tm)
library(e1071)
library(text2vec)
library(dplyr)

# We create now a sample data set with texts of philosophical or non-philosophical nature
IDs <- c(1:4)
Texts <- c("Ludwig Wittgenstein has significantly contributed to the development of Philosophy of Science.",
           "Neuroaesthetics studies the neural underpinnings of aesthetics.",
           "Data analysis is appealing for bioinformaticians and curious cognitive scientists.",
           "Eyes are the mirrors of the soul."
           )

# We create a csv file where the sample texts will be stored
sample_texts_df <- data.frame(
  ID = IDs,
  text = Texts,
  stringsAsFactors = FALSE
)

# If we want to have a look on the data
glimpse(sample_texts_df)

# If we want to print the data frame
print(sample_texts_df)

# Since some of the texts are philosophical and some are not, we need to label them for the 
# Naive Bayes classification. 

Labels <- factor(c(1, 1, 0, 1)) # the first two texts and the last one are of philosophical nature
                                # the third text is non-philosophical

# Now, the data will be pre-processed (before the application of the classification model)

# First, we will create a corpus of the sample texts
corpus <- VCorpus(VectorSource(sample_texts_df$text))

# Conversion to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))

# Removal of any present punctuation
corpus <- tm_map(corpus, removePunctuation)

# Stop-word removal
corpus <- tm_map(corpus, removeWords, stopwords("en"))

# Number removal (if present)
corpus <- tm_map(corpus, removeNumbers)

# White-space removal (if present)
corpus <- tm_map(corpus, stripWhitespace)

# After the pre-processing, we will convert our corpus to a document-term matrix (DTM)
DTM <- DocumentTermMatrix(corpus)
DTM_Matrix <- as.matrix(DTM)

# Now, we will train a Naive Bayes model to classify the texts (and any given output) as either 
# philosophical or non-philosophical
NB_classifier <- naiveBayes(DTM_Matrix, Labels)

# Now, we will provide novel texts and classify them in accordance to our model
novel_text <- "All science starts with a philosophical question."

# Pre-processing will be applied as earlier for the novel text (same procedure)
novel_corpus <- VCorpus(VectorSource(novel_text))
novel_corpus <- tm_map(novel_corpus, content_transformer(tolower))
novel_corpus <- tm_map(novel_corpus, removePunctuation)
novel_corpus <- tm_map(novel_corpus, removeWords, stopwords("en"))
novel_corpus <- tm_map(novel_corpus, removeNumbers)
novel_corpus <- tm_map(novel_corpus, stripWhitespace)

# Conversion to a document-term matrix (DTM)
novel_DTM <- DocumentTermMatrix(novel_corpus, control = list(dictionary = Terms(DTM)))

# Conversion to matrix
novel_DTM_Matrix <- as.matrix(novel_DTM)

# Now, we predict whether the text is philosophical or non-philosophical (based on the NB classifier)
text_prediction <- predict(NB_classifier, novel_DTM_Matrix)

# Print the output
print(text_prediction)

# Before we proceed to the visualization of our classification model, we first need to convert the printed output
# (text prediction) into a matrix format
text_prediction_matrix <- table(text_prediction)

# We print the text prediction matrix

print(text_prediction_matrix)

# We install the required package for the cosine similarity (and load the respective library)

install.packages("proxy")
library(proxy)

# Cosine similarity
cosine_sim <- simil(DTM_Matrix)  #cosine similarity between the texts of the sample_texts_df

# Viewing the cosine similarity matrix
print(cosine_sim)

# How to visualize the cosine similarity heat-map
heatmap(as.matrix(cosine_sim), main = "Cosine Similarity", col = heat.colors(256))

# If we were to differently visualize the classification (philosophical texts or non-philosophical texts),
# we would apply the following (bar-plot function):
barplot(text_prediction_matrix,
        main = "Classification as philosophical or non-philosophical",
        ylab = "Probability",
        col = c("blue", "red"),
        names.arg = c("Non-philosophical", "Philosophical"))

# For examining the cosine similarity between our texts (sample texts and novel text) 

# First, we install the required packages (and load the respective libraries)
install.packages("Matrix")
install.packages("textTinyR")

library(Matrix)
library(textTinyR)
library(proxy)

# Cosine similarity:
cosine_sim_2 <- simil(novel_DTM_Matrix, DTM_Matrix)

# Print the cosine similarity
print(cosine_sim_2)

# Visualizing the cosine similarity between the two matrices (bar-plot):
install.packages("gplots")
library(gplots)
 
barplot(cosine_sim_2[1, ],
        main = "Cosine Similarity",
        ylab = "Similarity",
        xlab = "Corpus Matrices",
        col = "red",
        names.arg = colnames(cosine_sim_2),
        las = 2)


# We will visualize with a pheat-map (since we have 1 row in our matrix)
install.packages("pheatmap")
library(pheatmap)

# Visualizing with a pheat-map:
pheatmap(as.matrix(cosine_sim_2), color = colorRampPalette(c("red", "black"))(100),
         cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Cosine Similarity")

############################################################################

# How about performing sentiment analysis of our sample texts?

# We first install the required packages (and we load the respective libraries)
install.packages("syuzhet")
library(syuzhet)

# We calculate the sentiment scores 
sentiment_scores <- get_nrc_sentiment(sample_texts_df$text)

# Printing the sentiment scores
print(sentiment_scores)

#Visualizing sentiments
barplot(
  sort(colSums(sentiment_scores)),
  las = 2,
  col = rainbow(2),
  main = "Sentiment Analysis"
)

# We may apply the same procedure for our novel text
sentiment_scores_2 <- get_nrc_sentiment(novel_text)

# Printing the sentiment scores
print(sentiment_scores_2)

# Visualizing sentiments
barplot(
  sort(colSums(sentiment_scores_2)),
  las = 2,
  col = rainbow(2),
  main = "Sentiment Analysis"
)

####################### End of data analysis ################################
#############################################################################


































