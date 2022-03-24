# Import libraries and modules needed
import nltk
import pandas as pd
import pickle
import os
import sys
import os.path
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

#filepath = os.path.abspath('sentiment-analysis-on-movie-reviews/train.tsv/train.tsv')

#filepath_test = os.path.abspath('sentiment-analysis-on-movie-reviews/test.tsv/test.tsv')

#vocabulary_path = 'model/vocab'

def vectorize_tsv_data(filepath, column_name, operation_mode, vocabulary_path, max_feat):
    """
    Reads data from the filepath creates the respective vectors
    and returns them
    """

    # If in test/predict mode we have to load the vocabulary from the disk.
    if operation_mode == 'predict':
        voc = pickle.load(open(vocabulary_path, 'rb'))
        print('vocabulary')
        print(voc)

    # Read from tsv file our data
    docs = pd.read_csv(filepath, sep='\t')
    # Load the specific tsv column
    data = docs[column_name]

    # It will filter symbols and numbers
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

    # Vectorize

    # This is for training
    if operation_mode == 'train':
        # Instantiate the vectorizer object
        # How the  vectorizer decides which vocabulary it will use for the feature vectors?
        # more frequent features -> sorted by name
        print('training')
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1),
                                     tokenizer=tokenizer.tokenize,
                                     max_features=max_feat)

    # This is for testing/predicting
    elif operation_mode == 'predict':
        # Instantiate the vectorizer object
        # pass vocabulary
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1),
                                     tokenizer=tokenizer.tokenize,
                                     max_features=max_feat, vocabulary=voc)
    else:
        print('not valid operation mode')

    # Convert the documents into a matrix
    vectors = vectorizer.fit_transform(data)

    if operation_mode == 'train':
        # save vocabulary
        # we will need it for testing
        pickle.dump(vectorizer.vocabulary_, open(vocabulary_path, 'wb'))

    return vectors



def get_column_from_tsv_data(filepath, column_name):
    """
    Load and return a column
    :param filepath:
    :param column_name:
    :return:
    """
    # Read from tsv our data
    docs = pd.read_csv(filepath, sep='\t')
    labels = docs[column_name]

    return labels
