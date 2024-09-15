# %% [markdown]
# ### Use a pretrained Word2vec model (Google news). Choose a short English text (about 400-500 words). For example you can take a wikipedia article or book excerpt. The text must also contain proper nouns. Solve the following tasks:

# %% [markdown]
# #### 1. Print the number of words in the model's vocabulary.

# %%
text = "Born into an upper-middle-class family, Van Gogh drew as a child and was serious, quiet and thoughtful, but showed signs of mental instability. As a young man, he worked as an art dealer, often travelling, but became depressed after he was transferred to London. He turned to religion and spent time as a missionary in southern Belgium. Later he drifted into ill-health and solitude. He was keenly aware of modernist trends in art and, while back with his parents, took up painting in 1881. His younger brother, Theo, supported him financially, and the two of them maintained a long correspondence."

# %%
from gensim.models import KeyedVectors

model_path = "GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

vocab = set(model.key_to_index)
print("Number of words in the model's vocabulary: ", len(vocab))

# %% [markdown]
# #### 2. Print all the words in the text that do not appear in the model's vocabulary.

# %%
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')


words = set(word_tokenize(text))

for word in words:
    if word not in vocab:
        print(word)

# %% [markdown]
# #### 3. Which are the two most distant words in the text, and which are the closest? Print the distance too.

# %%
from itertools import combinations
import numpy as np

word_pairs = list(combinations(words, 2))

most_distant_words = None
most_distant_words = None
max_distance = -np.inf
min_distance = np.inf

for word1, word2 in word_pairs:
    if word1 in vocab and word2 in vocab:
        distance = model.similarity(word1, word2)
        if distance > max_distance:
            most_similar_words = (word1, word2)
            max_distance = distance
        if distance < min_distance:
            most_distant_words = (word1, word2)
            min_distance = distance

# Print the most distant and the closest words and their distances
print(f"The most distant words are {most_distant_words} with a distance of {max_distance}")
print(f"The closest words are {most_similar_words} with a distance of {min_distance}")

# %% [markdown]
# #### 4. Using NER (Named Entity Recognition) find the named entities in the text. Print the first 5 most similar words to them both in upper and lowercase.

# %%
import spacy

#python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

for ent in doc.ents:
    if ent.text in model.key_to_index:
        print(f"Named entity: {ent.text}")
        similar_words_lower = model.most_similar(ent.text.lower(), topn=5)
        similar_words_upper = model.most_similar(ent.text, topn=5)
        for (word_lower, _), (word_upper, _) in zip(similar_words_lower, similar_words_upper):
            print(f"lowercase: {word_lower}\nuppercase: {word_upper}")
        print()

# %% [markdown]
# #### 5. Print the clusters of words that are the most similar in the text (you can use sklearn's Kmeans) based on their vectors in the model.

# %%
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize


vectors = [model[word] for word in words if word in vocab]

kmeans = KMeans(n_clusters=3).fit(vectors)
labels = kmeans.labels_

clusters = {i: [] for i in range(kmeans.n_clusters)}

for word, label in zip(words, labels):
    clusters[label].append(word)

for label, words in clusters.items():
    print(f"Cluster {label}: {words}")


