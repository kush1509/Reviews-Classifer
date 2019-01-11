# library to clean data
import re

# Natural Language Tool Kit
import nltk
from sklearn import svm
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# to remove stopword
from nltk.corpus import stopwords

# for Stemming propose
from nltk.stem.porter import PorterStemmer

# Initialize empty array
# to append clean text
corpus = []

# Importing Libraries
import pandas as pd

# Import dataset
dataset = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter = '\t')


# 1000 (reviews) rows to clean
for i in range(0, 1000):
    # column : "Review", row ith
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    # convert all cases to lower cases
    review = review.lower()

    # split to array(default delimiter is " ")
    review = review.split()

    # creating PorterStemmer object to
    # take main stem of each word
    ps = PorterStemmer()

    # loop for stemming each word
    # in string array at ith row
    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]

    # rejoin all string array elements
    # to create back into a string
    review = ' '.join(review)

    # append each string to create
    # array of clean text
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# To extract max 1500 feature.
# "max_features" is attribute to
# experiment with to get better results
tfidf  = TfidfVectorizer()

# X contains corpus (dependent variable)
X = tfidf.fit_transform(corpus).toarray()

# y contains answers if review
# is positive or negative
y = dataset.iloc[:, 1].values

num_training = 750
X_train = X[:num_training]
X_test = X[num_training:]
y_train = dataset.iloc[:, 1].values[:num_training]
y_test = dataset.iloc[:, 1].values[num_training:]

# print(X_train[:, 0].size)

model = svm.SVC(C=10, kernel='linear')
# model = svm.LinearSVC(C=)
print("Training model...")
model.fit(X_train, y_train)

print("Running predictions...")

predictions = model.predict(X_test)

print ("FINISHED classifying. accuracy score : ")
print (accuracy_score(y_test, predictions))