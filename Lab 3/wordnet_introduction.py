# %% [markdown]
# ### 1. Create a function that receives a word and prints the associated glosses for all the possible senses of that word (you must find all its corresponding synsets and print the gloss for each)

# %%
import nltk
from nltk.corpus import wordnet

# nltk.download('wordnet', quiet=True)  

def print_glosses(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        print(f"Sense: {synset.name()}")
        print(f"Gloss: {synset.definition()}")
        print()

print_glosses('bank')

# %% [markdown]
# ### 2. Create a function that receives two words as parameters. The function will check, using WordNet if the two words can be synonyms (there is at least one synset that contains the two words). If such synset is found, rint the gloss for that synset.

# %%
import nltk
from nltk.corpus import wordnet

def check_synonyms(word1, word2):
    
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 == synset2:
                print(f"Synset: {synset1.name()}")
                print(f"Gloss: {synset1.definition()}")
                return True
    return False

check_synonyms('happy', 'glad')

# %% [markdown]
# ### 3. Create a function that receives a synset object and returns a tuple with 2 lists. The first list contains the holonyms (all types of holonyms) and the second one the meronyms (all types). Find a word that has either holonyms or meronyms of different types. Print them separately (on cathegories of holonyms/meronyms) and then all together using the created function (in order to check that it prints them all).

# %%
def get_holonyms_meronyms(synset):
    holonyms = synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms()
    meronyms = synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms()
    return (holonyms, meronyms)

tree_synset = wordnet.synset('tree.n.01')

print("Member holonyms:", tree_synset.member_holonyms())
print("Part holonyms:", tree_synset.part_holonyms())
print("Substance holonyms:", tree_synset.substance_holonyms())
print("Member meronyms:", tree_synset.member_meronyms())
print("Part meronyms:", tree_synset.part_meronyms())
print("Substance meronyms:", tree_synset.substance_meronyms())

holonyms, meronyms = get_holonyms_meronyms(tree_synset)
print("All holonyms:", holonyms)
print("All meronyms:", meronyms)

# %% [markdown]
# ### 4. Create a function that for a given synset, prints the path of hypernyms (going to the next hypernym, and from that hypernym to the next one and so on, until it reaches the root).

# %%
def print_hypernym_path(synset):
    while synset:
        print(synset.name())
        if synset.hypernyms():
            synset = synset.hypernyms()[0]
        else:
            break

print_hypernym_path(wordnet.synset('tree.n.01'))

# %% [markdown]
# ### 5. Create a function that receives two synsets as parameters. We consider d1(k) the length of the path from the first word to the hypernym k (the length of the path is the number of hypernyms it goes through, to reach k) and d2(k) the length of the path from the second word to the hypernym k. The function will return the list of hypernyms having the property that d1(k)+d2(k) is minimum

# %%
def find_common_hypernyms(synset1, synset2):

    paths1 = synset1.hypernym_paths()
    paths2 = synset2.hypernym_paths()

    min_sum = float('inf')

    common_hypernyms = []

    for path1 in paths1:
        for path2 in paths2:

            common = set(path1).intersection(path2)

            for hypernym in common:

                sum_lengths = path1.index(hypernym) + path2.index(hypernym)

                if sum_lengths < min_sum:
                    min_sum = sum_lengths
                    common_hypernyms = [hypernym]
                elif sum_lengths == min_sum:
                    common_hypernyms.append(hypernym)

    return set(common_hypernyms)


synset1 = wordnet.synset('car.n.01')
synset2 = wordnet.synset('bus.n.01')
print(find_common_hypernyms(synset1, synset2))

# %% [markdown]
# ### 6. Create a function that receives a synset object and a list of synsets (the list must contain at least 5 elements). The function will return a sorted list. The list will be sorted by the similarity between the first synset and the synsets in the list. For example (we consider we take the firs synset for each word) we can test for the word cat and the list: animal, tree, house, object, public_school, mouse.

# %%
def sort_by_similarity(synset, synsets):

    similarities = [(other_synset, synset.path_similarity(other_synset)) for other_synset in synsets]
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [synset for synset, similarity in similarities]


synset = wordnet.synset('cat.n.01')
synsets = [wordnet.synset(word + '.n.01') for word in ['animal', 'tree', 'house', 'object', 'public_school', 'mouse']]
sorted_synsets = sort_by_similarity(synset, synsets)

for synset in sorted_synsets:
    print(synset.name())

# %% [markdown]
# ### 7. Create a function that checks if two synsets can be indirect meronyms for the same synset. An indirect meronym is either a part of the givem element or a part of a part of the given element (and we can exten this relation as being part of part of part of etc....). This applies to any type of meronym.

# %%
def check_indirect_meronyms(synset1, synset2):

    all_synsets = list(wordnet.all_synsets())

    for synset in all_synsets:

        meronyms = set()
        to_check = [synset]
        while to_check:
            current_synset = to_check.pop()
            current_meronyms = current_synset.member_meronyms() + current_synset.part_meronyms() + current_synset.substance_meronyms()
            for meronym in current_meronyms:
                if meronym not in meronyms:
                    meronyms.add(meronym)
                    to_check.append(meronym)

        if synset1 in meronyms and synset2 in meronyms:
            return True

    return False


synset1 = wordnet.synset('nose.n.01')
synset2 = wordnet.synset('tail.n.01')
print(check_indirect_meronyms(synset1, synset2))

# %% [markdown]
# ### 8. Print the synonyms and antonyms of an adjective (for example, "beautiful"). If it's polisemantic, print them for each sense, also printing the gloss for that sense (synset).

# %%
from nltk.corpus import wordnet

def print_synonyms_antonyms(word):
    for synset in wordnet.synsets(word, pos=wordnet.ADJ):
        print(f"Sense: {synset.name()}")
        print(f"Gloss: {synset.definition()}")
        print("Synonyms:", ", ".join([lemma.name() for lemma in synset.lemmas()]))
        antonyms = []
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.extend([antonym.name() for antonym in lemma.antonyms()])
        print("Antonyms:", ", ".join(antonyms))
        print()

print_synonyms_antonyms("beautiful")


