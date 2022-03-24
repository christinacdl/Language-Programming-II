
# ================================================================== EXERCISE 03 ================================================================================
# Import libraries and modules needed
import nltk
import os
import os.path
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.sparse import csr_matrix
from nltk.collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from dataset1 import vectorize_tsv_data, get_column_from_tsv_data

# We split the train.tsv into new tsv files named train_split.tsv and test1.tsv using split = 10100 (split_tsv.py)
# For every n-feature(100,1000,5000,10000) we changed the value of max_feat in vectorize_tsv_data function and the model name to get the results

# Get the filepath of the train.tsv
filepath = os.path.abspath('sentiment-analysis-on-movie-reviews/train.tsv/train_split.tsv')

# Get the filepath of the test.tsv
filepath_test = os.path.abspath('sentiment-analysis-on-movie-reviews/test.tsv/test1.tsv')

# Get the vocabulary path
vocabulary_path = os.path.abspath('model/vocab')

# Create vectors in train operation mode and get n-features using the vectorize_tsv_data from dataset1
vectors = vectorize_tsv_data(filepath, 'Phrase', 'train', vocabulary_path, 10000)

# Transform the sparse matrix of the train set into a dense matrix using to.dense function
dense_matrix = vectors.todense()

# Load the column 'Phrase' from the filepath using the get_column_from_tsv_data function from dataset1
column_name = get_column_from_tsv_data(filepath,'Phrase')

# Create a function that returns new vectors with two POS features
def utils(dense_matrix, filepath):
    dataframe = pd.read_table(filepath)
    df = dataframe['Phrase'].tolist()

    # Join matrix Initialization that has a shape of number of rows (len(df)) and 2 columns (adj, adv)
    join = np.zeros((len(df), 2))
    # For every row in the df tokenize
    tokens = [nltk.word_tokenize(row) for row in df]

    # Enumerate in order to have the index (row number of every line processed)
    for index, token in enumerate(tokens):
        ud_tags = nltk.pos_tag(token, 'universal')
        ud = {'ADV', 'ADJ'}
        just_pos = []
        # Append all the tags that match 'ADV', 'ADJ' to just_pos
        for tag in ud_tags:
            if tag[1] in ud:
                just_pos.append(tag[1])

        # Count the 'ADV', 'ADJ' after appending all the ud_tags of the row
        counts = Counter(just_pos)
        adj = counts['ADJ']
        adv = counts['ADV']

        # Update join at the respective index with the 'ADJ' and 'ADV' values
        join[index][0] = adj
        join[index][1] = adv

        # Return the Horizontal Stack of dense_matrix and join
        matrix = np.hstack((dense_matrix, join))

        # Transform and print the dense matrix into a sparse matrix using csr_matrix module from scipy.sparse library
        new_vectors = csr_matrix(matrix)
        #print('The new feature vectors:\n',new_vectors)
        return new_vectors

# Save the utils function into final_matrix variable
final_matrix = utils(dense_matrix, filepath)

# Print the dimensions of the final_matrix
print(final_matrix.shape)

#------------------------------------------------------------Train------------------------------------------------------------------------------------------
#feature_vectors = vectorize_tsv_data(filepath, 'Phrase', 'train',vocabulary_path, 100)
feature_labels = get_column_from_tsv_data(filepath, 'Sentiment')

def train(feature_vectors,feature_labels):

    knn_model = KNeighborsClassifier(n_neighbors = 5 ).fit(feature_vectors, feature_labels)

    # Save the knn.model to disk using pickle with its corresponding size of feature vectors
    filename = './model/knn.model' + str(feature_vectors.shape[1])
    print(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(knn_model, open(filename, 'wb'))

# Run train function using the matrix of the train set
train(final_matrix,feature_labels)

#-----------------------------------------------------------Predict-----------------------------------------------------------------------------------------
# Create vectors in predict operation mode and get n-features using the vectorize_tsv_data from dataset1
test_vectors = vectorize_tsv_data(filepath_test, 'Phrase', 'predict', vocabulary_path, 10000)

# Transform the sparse matrix of the test_vectors into a dense matrix using to.dense function
test_dense_matrix = test_vectors.todense()

# Use the utils function for the test set
test_final_matrix = utils(test_dense_matrix,filepath_test)

# Load knn.model
filename = './model/knn.model10002'
loaded_model = pickle.load(open(filename, 'rb'))
print('model loaded')

def predict(feature_vectors,filepath_test):

    # Predict
    print('predict')
    predictions = loaded_model.predict(feature_vectors)

    # Present results
    # Get phrases
    phrases = get_column_from_tsv_data(filepath_test, 'Phrase')
    length = len(phrases)

    result = open('results.txt', 'w')
    # Print Prediction and Phrase side by side
    print("{0:5}{1:15}{2:200}".format("Num","Prediction","Phrase"), file =result)
    print('------------------------------------------------------------------------------------------------------------')
    for i in range(length):
        print ("{0:5}{1:15}{2:200}".format(str(i), str(predictions[i]), phrases[i]), file =result)

    #-------- Evaluate ---------

    # Get correct labels
    correct_labels = get_column_from_tsv_data(filepath_test, 'Sentiment')

    # Calculate accuracy using the metrics module from sklearn
    acc = metrics.accuracy_score(correct_labels, predictions)
    print("Accuracy:", acc)

    # Create a confusion matrix
    matrix = metrics.confusion_matrix(correct_labels, predictions)
    print(matrix)

    # Calculate precision, recall and fscore using the metrics module from sklearn
    precision = metrics.precision_score(correct_labels, predictions, average=None)
    recall = metrics.recall_score(correct_labels, predictions, average=None)
    fscore = metrics.f1_score(correct_labels, predictions, average=None)


    print("{0:30}{1:30}{2:30}{3:30}".format("Class","Precision","Recall","F-measure"))
    print('------------------------------------------------------------------------------------------------------------')
    for i in range(len(precision)):
        print("{0:30}{1:30}{2:30}{3:30}".format(str(i), str(precision[i]), str(recall[i]), str(fscore[i])))

# Run predict function
predict(test_final_matrix,filepath_test)

