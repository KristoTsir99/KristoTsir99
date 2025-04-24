# This code is a python version of the r and matlab code conducted on NLP. 
# The code was adapted from ChatGPT
# For further information, you may visit https://openai.com/chatgpt

# As a first step, we import the required packages/modules 

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Then, we download the required resources for NLP (stopwords, punctuation, etc.)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#  We store the sample texts in a variable
text_1 = 'Ανακαλύπτουμε οι δυο μας τη ζωή, τις αλλαγές κοιτάμε σκυθρωποί.'
text_2 = 'Εμβρόντητος φωνάζεις τη σκέψη τη ζεστή, σαν χάδι από μετάξι να έρθει να σε αρπάξει.'

# Text preprocessing: Conversion to lowercase, tokenization, and stopword removal
stop_words = set(stopwords.words('greek'))


# Tokenization and stopword removal
# Text 1
def preprocess_text(text_1):
    tokens = word_tokenize(text_1.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Text 2
    tokens_2 = word_tokenize(text_2.lower())
    tokens_2 = [word for word in tokens_2 if word.isalpha() and word not in stop_words]

# Text preprocessing
text_1_preprocessed = preprocess_text(text_1)
text_2_preprocessed = preprocess_text(text_2)

# Showing the preprocessing results

print(f"Preprocessed Text 1: {text_1_preprocessed}")
print(f"Preprocessed Text 2: {text_2_preprocessed}")

# Term-Document Matrix (DTM), using the vectorizer function
Vectorizer = CountVectorizer()

# Combining the texts into a list
sample_texts = [text_1_preprocessed, text_2_preprocessed]

# Transforming the sample texts into a term-document matrix
DTM = Vectorizer.fit_transform(sample_texts)

# Getting the words in the DTM
Words = Vectorizer.get_feature_names_out()

# Conversion of the DTM matrix into an array
DTM_array = DTM.toarray()

# Show the DTM
print(f"\nTerm-Document Matrix:\n{DTM}")
print(f"Words: {Words}")
print(f"TDM (Documents x Words):\n{DTM_array}")

# Cosine similarity between the two texts
cosine_sim_sample_texts = cosine_similarity(DTM_array)

# Show the cosine similarity matrix
print(f"\nCosine Similarity Matrix:\n{cosine_sim_sample_texts}")

# Visualizing the cosine similarity matrix with a heatmap 
plt.figure(figsize=(5, 4))
sns.heatmap(cosine_sim_sample_texts, annot=True, cmap='Blues', xticklabels=["Text 1", "Text 2"], yticklabels=["Text 1", "Text 2"])
plt.title("Cosine Similarity")
plt.show()

# Now, we will be creating a bar plot of the cosine similarity

# First, we need to import some packages/modules 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

 # First, we will use the vectorizer function
 # for term frequency–inverse document frequency (feature extraction)
vectorizer_1 = TfidfVectorizer()

# Then, we will create a matrix
tfidf_matrix = vectorizer_1.fit_transform(sample_texts)

# Then, we will create a cosine similarity matrix of the tfidf matrix
cos_sim_matrix = cosine_similarity(tfidf_matrix)

# Similarity between the sample texts
similarity_value = cos_sim_matrix[0, 1]

# For the bar plot, we will need to make our two-size array to one-size array
# First way
arr = np.array(cos_sim_matrix)
flattened = arr.flatten()

# Second way
flattened = arr.reshape(-1)

# Showing the output
print(flattened)

# Bar plot
plt.bar(range(len(flattened)), flattened)
plt.title("All Cosine Similarity Values (Flattened)")
plt.ylim([0, 1])
plt.show()