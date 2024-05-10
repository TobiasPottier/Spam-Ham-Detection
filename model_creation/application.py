import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import nltk
from nltk.corpus import stopwords
import re #regular expressions
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
import os
import csv
from joblib import dump

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_colwidth',150)

# Import dataset and split in features and targets

# read dataset
dataset = pd.read_csv('spam_train.csv')

testset = pd.read_csv('spam_test.csv')

# Split features from targets
y_train = dataset.type.values
X_train = dataset.text.values

y_test = testset.type.values
X_test = testset.text.values

testset.tail(20)
from imblearn.over_sampling import SMOTE


# Text preprocessing

def text_preprocessing(text, language, minWordSize):
    
    # remove html
    text_no_html = BeautifulSoup(str(text),features="html.parser").get_text()
    
    # remove non-letters
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(text_no_html)) 
        
    # convert to lower-case
    text_lower = text_alpha_chars.lower()
    
    # remove stop words
    stops = set(stopwords.words(language)) 
    text_no_stop_words = ' '
    
    for w in text_lower.split():
        if w not in stops:  
            text_no_stop_words = text_no_stop_words + w + ' '
      
       # do stemming
    text_stemmer = ' '
    stemmer = SnowballStemmer(language)
    for w in text_no_stop_words.split():
        text_stemmer = text_stemmer + stemmer.stem(w) + ' '
         
    # remove short words
    text_no_short_words = ' '
    for w in text_stemmer.split(): 
        if len(w) >= minWordSize:
            text_no_short_words = text_no_short_words + w + ' '
 

    return text_no_short_words

# Convert training and test set to bag of words
language = 'english'
minWordLength = 2

for i in range(X_train.size):
    X_train[i] = text_preprocessing(X_train[i], language, minWordLength)
    
    
for i in range(X_test.size):
    X_test[i] = text_preprocessing(X_test[i], language, minWordLength)
    
# Make sparse features vectors 
# Bag of words

count_vect = CountVectorizer()
X_train_bag_of_words = count_vect.fit(X_train)
X_train_bag_of_words = count_vect.transform(X_train)
X_test_bag_of_words = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_bag_of_words)
X_train_tf = tf_transformer.transform(X_train_bag_of_words)
X_test_tf = tf_transformer.transform(X_test_bag_of_words)

smote = SMOTE()
X_train_tf, y_train = smote.fit_resample(X_train_tf, y_train)
X_test_tf, y_test = smote.fit_resample(X_test_tf, y_test)

def print_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred) * 100)

# Naive bayes
NBclassifier = MultinomialNB(alpha=1)
NBclassifier.fit(X_train_tf, y_train)
y_pred_model_nb = NBclassifier.predict(X_test_tf)

# train a logistic regression classifier
lregclassifier = LogisticRegression(C=10, class_weight='balanced')
lregclassifier.fit(X_train_tf, y_train)
y_pred_model_lr = lregclassifier.predict(X_test_tf)


dump(NBclassifier, 'NBclassifier.joblib')
dump(lregclassifier, 'lregclassifier.joblib')
dump(count_vect, 'count_vect.joblib')
dump(tf_transformer, 'tf_transformer.joblib')