
# ============================================================= EXERCISE 02 ================================================================================
# Import libraries and modules needed
import nltk
import os
import os.path
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from operator import itemgetter, attrgetter
from itertools import zip_longest


text_file = os.path.abspath("corpus/How Princess Diana has been portrayed on stage and in film.txt")  # Finds the absolute path of the text
#print(text_file)  # Prints the full path of the file


load_file = open(text_file, "r", encoding="utf8")  # Opens the file for reading with encoding UTF-8
read_file = load_file.read()  # Reads all the lines of the file
#print(read_file)  # Prints the full text
#print(f'Length of text:',len(read_file))
load_file.close()  # Closes the file as necessary


# Splits the text with word_tokenize() offered by nltk
words = nltk.word_tokenize(read_file)
#print(words)


# Filters out punctuation
tokens = [token for token in words if token.isalpha()]  # the function isalpha() removes the non-alphabetic tokens
#print(f'Tokens without punctuation for every text in the folder:\n',tokens[:100])


# Uses porter stemmer
porter = PorterStemmer()
porter_stem = [porter.stem(token) for token in tokens]  # for every token in the list of tokens applies PorterStemmer
#print(f'Stemmed tokens with PorterStemmer:', tokens1)


# Uses lancaster stemmer
lancaster = LancasterStemmer()
lancaster_stem = [lancaster.stem(token) for token in tokens]  # for every token in the list of tokens applies LancasterStemmer
#print(f'Stemmed tokens with LancasterStemmer:',tokens)


# Merges and then prints word side by side the porter stem and the lancaster stem
merge_word_with_stems = {tokens: (porter_stem, lancaster_stem) for tokens, porter_stem, lancaster_stem in zip_longest(tokens, porter_stem, lancaster_stem)}
print(f'Merging word with porter stem & lancaster stem respectively: \n', merge_word_with_stems)
