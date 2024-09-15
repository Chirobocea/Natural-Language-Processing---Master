#!/usr/bin/env python
# coding: utf-8

# ### Ex 1
# 
# Download it through python (inside the code, so you don't have to upload the file too when you send the solution for this exercise) with urlopen() from module urllib and read the entire text in one single string. If the download takes too much time at each running, download the file, but leave the former instructions in a comment (to show that you know how to access an online file)

# In[1]:


import urllib.request

url = "https://www.gutenberg.org/cache/epub/73288/pg73288.txt"

response = urllib.request.urlopen(url)
text = response.read().decode("utf-8") 
print("Text downloaded successfully!")


# ### Ex 2
# 
# Remove the header (keep only the text starting from the title)

# In[10]:


header_treshold = "*** START OF THE PROJECT GUTENBERG EBOOK THE SURVIVORS ***"
title_index = text.find(header_treshold)
text_without_header = text[title_index+len(header_treshold):]


# ### Ex 3
# 
# Print the number of sentences in the text. Print the average length (number of words) of a sentence.

# In[12]:


import re

sentences = re.split(r'[.!?]', text_without_header)

sentences = [sentence for sentence in sentences if sentence.strip()]

num_sentences = len(sentences)

avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / num_sentences

print(f"Number of sentences: {num_sentences}")
print(f"Average length of a sentence: {avg_sentence_length} words")


# ### Ex 4
# 
# Find the collocations in the text (bigram and trigram). Use the nltk.collocations module You will print them only once not each time they appear.

# In[17]:


import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

nltk.download('punkt')

tokens = nltk.word_tokenize(text_without_header)

collocations_treshold = 10

bigram_finder = BigramCollocationFinder.from_words(tokens)
bigram_collocations = set(bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, collocations_treshold))

trigram_finder = TrigramCollocationFinder.from_words(tokens)
trigram_collocations = set(trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, collocations_treshold))

print("Bigram Collocations:")
for bigram in bigram_collocations:
    print(bigram)

print("\nTrigram Collocations:")
for trigram in trigram_collocations:
    print(trigram)


# ### Ex 5
# Create a list of all the words (in lower case) from the text, without the punctuation.

# In[61]:


import string

text_without_punctuation = text_without_header.translate(str.maketrans('', '', string.punctuation)).lower()

words = nltk.word_tokenize(text_without_punctuation)
print(words)


# ### Ex 6
# Print the first N most frequent words (alphanumeric strings) together with their number of appearances.

# In[62]:


from collections import Counter

word_counts = Counter(words)

N = 10

most_common_words = word_counts.most_common(N)

for word, count in most_common_words:
    print(f"{word}: {count}")


# ### Ex 7
# Remove stopwords and assign the result to variable lws

# In[63]:


from nltk.corpus import stopwords

nltk.download('stopwords')

stopwords_set = set(stopwords.words('english'))

lws = [word for word in words if word not in stopwords_set]

N = 10

print(lws[:N])


# ### Ex 8
# Apply stemming (Porter) on the list of words (lws). Print the first 200 words. Do you see any words that don't appear in the dictionary?

# In[64]:


from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stemmed_words = list(set([stemmer.stem(word) for word in lws]))

print(stemmed_words[:200])


# ### Ex 9
# Print a table of three columns (of size N, where N is the maximum length for the words in the text). The columns will be separated with the character "|". The head of the table will be:
# 
# Porter    |Lancaster |Snowball
# 
# The table will contain only the words that give different stemming results for the three stemmers (for example, suppose that we have both "runs" and "being" inside the text. The word "runs" should not appear in the list, as all three results are "run"; however "being" should appear in the table). The stemming result for the word for each stemmer will appear in the table according to the head of the table. The table will contain the results for the first NW words from the text (the number of rows will obviously be less than NW, as not all words match the requirements). For example, NW=500. Try to print only distinct results inside the table (for example, if a word has two occurnces inside the text, and matches the requirments for appearing in the table, it should have only one corresponding row).

# In[65]:


from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer('english')

table = []
NW = 500


for word in words[:NW]:

    porter_stem = porter_stemmer.stem(word)
    lancaster_stem = lancaster_stemmer.stem(word)
    snowball_stem = snowball_stemmer.stem(word)
    
    # Check if the stemming results are different for each stemmer
    if porter_stem != lancaster_stem != snowball_stem:
        # Add the word and its stemming results to the table
        table.append([porter_stem, lancaster_stem, snowball_stem])

print("Porter    |Lancaster |Snowball")
for row in table:
    print(f"{row[0]:<10}|{row[1]:<10}|{row[2]:<10}")


# ### Ex 10
# Print a table of two columns, simillar to the one above, that will compare the results of stemming and lemmatization. The head of the table will contain the values: "Snowball" and "WordNetLemmatizer". The table must contain only words that give different results in the process of stemming and lemmatization (for example, the word "running"). The table will contain the results for the first NW words from the text (the number of rows will obviously be less than NW, as not all words match the requirements). For example, NW=500. Try to print only distinct results inside the table (for example, if a word has two occurnces inside the text, and matches the requirments for appearing in the table, it should have only one corresponding row).

# In[68]:


from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate

nltk.download('wordnet')

snowball_stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

stemmed_words = []
lemmatized_words = []

for word in tokens[:NW]:
    stemmed_word = snowball_stemmer.stem(word.lower())
    lemmatized_word = lemmatizer.lemmatize(word.lower())
    
    if stemmed_word != lemmatized_word:
        stemmed_words.append(stemmed_word)
        lemmatized_words.append(lemmatized_word)

table_data = list(zip(stemmed_words, lemmatized_words))
table_headers = ["Snowball", "WordNetLemmatizer"]

table = tabulate(table_data, headers=table_headers, tablefmt="grid")
print(table)


# ### Ex 11
# Print the first N most frequent lemmas (after the removal of stopwords) together with their number of appearances.

# In[75]:


lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in tokens:

    lemmatized_word = lemmatizer.lemmatize(word.lower())
    lemmatized_words.append(lemmatized_word)

lemma_word_counts = Counter(words)

N = 10

filtered_word_counts = {word: count for word, count in lemma_word_counts.items() if word not in stopwords_set}

sorted_word_counts = sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)

for lemma, count in sorted_word_counts[:N]:
    print(f"{lemma}: {count}")


# ### Ex 12
# Change all the numbers from lws into words. Print the number of changes, and also the portion of list that contains first N changes (for example N=10).

# In[70]:


from num2words import num2words

lws_copy = lws.copy()
for i in range(len(lws_copy)):
    try:
        lws_copy[i] = num2words(int(lws_copy[i]))
    except:
        pass

differences = [lws_copy[i] for i in range(len(lws_copy)) if lws_copy[i] != lws[i]]

N = 10
print("Top", N, "differences:")
print(differences[:N])


# ### Ex 13
# Create a function that receives an integer N and a word W as parameter (it can also receive the list of words from the text). We want to print the concordance data for that word. This means printing the window of text (words on consecutive positions) of length N, that has the givend word W in the middle. For example, for the text ""I have two dogs and a cat. Do you have pets too? My cat likes to chase mice. My dogs like to chase my cat." and a window of length 3, the concordance data for the word "cat" would be ["dogs", "cat", "pets"] and ["pets","cat", "likes"] (we consider the text without stopwords and punctuation). However, as you can see, the window of text may contain words from different sentences. Create a second function that prints windows of texts that contain words only from the phrase containing word W. We want to print concordance data for all the inflexions of word W.

# In[73]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def get_concordance_data(N, W, text):

    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    
    # Find the base form of the target word
    base_W = lemmatizer.lemmatize(W)

    # Find the concordance data for all inflections of word W
    concordance_data = {}
    for i in range(len(tokens)):
        # Find the base form of the current word
        base_word = lemmatizer.lemmatize(tokens[i])

        # If the base form of the current word is the same as the base form of the target word
        if base_word == base_W:
            start_index = max(0, i - N)
            end_index = min(len(tokens), i + N + 1)
            window = tokens[start_index:end_index]

            # If the current word is not already in the concordance data, add it
            if tokens[i] not in concordance_data:
                concordance_data[tokens[i]] = []

            concordance_data[tokens[i]].append(window)
    
    return concordance_data

def get_phrase_concordance_data(N, W, text):

    lemmatizer = WordNetLemmatizer()

    sentences = nltk.sent_tokenize(text)
    
    base_W = lemmatizer.lemmatize(W)

    concordance_data = {}
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english') and token not in string.punctuation]
        
        for i in range(len(tokens)):
            # Find the base form of the current word
            base_word = lemmatizer.lemmatize(tokens[i])

            # If the base form of the current word is the same as the base form of the target word
            if base_word == base_W:
                start_index = max(0, i - N)
                end_index = min(len(tokens), i + N + 1)
                window = tokens[start_index:end_index]

                # If the current word is not already in the concordance data, add it
                if tokens[i] not in concordance_data:
                    concordance_data[tokens[i]] = []

                concordance_data[tokens[i]].append(window)
    
    return concordance_data

text = "I have two dogs and a cat. Do you have pets too? My cat likes to chase mice. My dogs like to chase my cat."
N = 3
W = "cat"

concordance_data = get_concordance_data(N, W, text)
print("Concordance data for all inflections of word", W)
for word, windows in concordance_data.items():
    print("\nConcordance data for word", word)
    for window in windows:
        print(window)

phrase_concordance_data = get_phrase_concordance_data(N, W, text)
print("\nPhrase concordance data for all inflections of word", W)
for word, windows in phrase_concordance_data.items():
    print("\nPhrase concordance data for word", word)
    for window in windows:
        print(window)

