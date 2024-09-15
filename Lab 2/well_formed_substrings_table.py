# %%
import nltk

# Define the grammar
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> Det N
  VP -> V NP
  Det -> 'the' | 'a'
  N -> 'cat' | 'dog'
  V -> 'chases' | 'sees'
""")

def print_wfst(sentence):

    sentence = sentence.split()

    # Initialize the WFST
    n = len(sentence)
    wfst = [[None for _ in range(n + 1)] for _ in range(n + 1)]

    # Populate the WFST for substrings of length 1
    for i in range(n):
        productions = grammar.productions(rhs=sentence[i])
        if productions:
            wfst[i][i+1] = productions[0].lhs()

    # Populate the WFST for substrings of length > 1
    for span in range(2, n + 1):
        for start in range(n + 1 - span):
            end = start + span
            for mid in range(start + 1, end):
                left_cell = wfst[start][mid]
                right_cell = wfst[mid][end]
                if left_cell and right_cell:
                    for prod in grammar.productions():
                        if prod.rhs() == (left_cell, right_cell):
                            wfst[start][end] = prod.lhs()
                            break

    for row in wfst:
        print(row)

print_wfst("the cat sees a dog")


