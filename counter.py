
# ========================================================== EXERCISE 01 ================================================================================
# Import libraries and modules needed
import os
import os.path
import nltk
import collections
from nltk.collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


corpus_path = os.path.abspath("corpus")  # Returns the full path of the folder named "corpus"
print(corpus_path)


stop_words_set = set(stopwords.words("english"))  # Creates a set from the english stopword list


for file in os.listdir(corpus_path):  # For every file that ends with .txt in the specific folder
    if file.endswith(".txt"):
        filename = os.path.join(corpus_path, file)  # Generates full path of each file in the folder
        #print(filename)   # Prints the full path of every text in the folder "corpus"
        file = open(filename, "r", encoding="utf8")  # Opens the file for reading with encoding UTF-8
        read_lines = file.read()  # Reads all the lines of the file
        file.close()  # Closes the file as necessary


        # Splits the texts with word_tokenize() offered by nltk
        tokenized_texts = nltk.word_tokenize(read_lines)


        # Filters out punctuation
        words = [word for word in tokenized_texts if word.isalpha()]  # the function isalpha() removes the non-alphabetic tokens
        #print(f'Tokens without punctuation for every text in the folder:\n',words[:70])


        # Filters out stopwords
        words = [w for w in words if not w in stop_words_set]
        #print(f'Tokens without stop_words for every text in the folder:\n',words[:70])


        # Finds the bigrams in each filtered text
        bigrams = list(nltk.bigrams(words))


        # Gets the 50 most frequent bigrams in every text in the English corpus
        bigram_freq = collections.Counter(bigrams)  # Counts the frequency of each bigram
        print(f'The 50 most frequent bigrams of {filename} :\n', bigram_freq.most_common(50))  # Uses f'string() method to print

