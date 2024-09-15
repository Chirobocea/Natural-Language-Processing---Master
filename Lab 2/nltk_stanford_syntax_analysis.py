# %%
import os
os.environ['STANFORD_PARSER'] = "stanford-parser-4.2.0/stanford-parser-full-2020-11-17/stanford-parser.jar"
os.environ['STANFORD_MODELS'] = "stanford-parser-4.2.0/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar"

from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tokenize import sent_tokenize


stanford_parser_dir = 'stanford-parser-4.2.0/stanford-parser-full-2020-11-17'
eng_model_path = stanford_parser_dir + "/stanford-parser-4.2.0-models/edu/models/lexparser/englishPCFG.ser.gz"
my_path_to_models_jar = stanford_parser_dir + "/stanford-parser-4.2.0-models.jar"
my_path_to_jar = stanford_parser_dir + "/stanford-parser.jar"

dep_parser = StanfordDependencyParser(path_to_jar=my_path_to_jar, path_to_models_jar=my_path_to_models_jar)
const_parser = StanfordParser(model_path=eng_model_path, java_options='-mx2g')

filename = 'sentences.txt'

# Create a file with some example sentences if it doesn't exist
if not os.path.isfile(filename):

    lines = ['The sun is shining brightly in the clear blue sky.',
            'I enjoy reading books and exploring new ideas.',
            'She walked along the beach, feeling the sand between her toes.',
            'The cat lazily stretched out on the warm laptop.',
            'We gathered around the campfire, sharing stories about Dracula.',
            'Let me know if you want something from the store.']

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

with open(filename, 'r') as f:
    text = f.read()
    
sentences = sent_tokenize(text)

# Parse the sentences and write the results to the output file
with open('parsed_sentences.txt', 'w') as f:
    for i, sentence in enumerate(sentences, start=1):
        f.write(f"Sentence - number {i}\n")
        f.write(f"{sentence}\n")

        # Constituency parsing
        const_tree = list(const_parser.raw_parse(sentence))[0]
        f.write(f"{const_tree}\n")

        # Dependency parsing
        dep_tree = list(dep_parser.raw_parse(sentence))[0]
        f.write(f"{dep_tree}\n")

        f.write("---------------------------------------------\n")


