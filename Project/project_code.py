import numpy as np
import pandas as pd
import csv

#import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input,  Activation
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, optimizers, layers
from sklearn.metrics import roc_auc_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
print(stopwords.words('english'))
import re                                  # library for regular expression operations
import string                              # for string operations
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import regexp_tokenize   # module for tokenizing strings
from nltk.tokenize import TreebankWordTokenizer

from google.colab import drive
drive.mount('/content/gdrive')


#importing the dataset
train=pd.read_csv('/content/gdrive/My Drive/CptS315Project/train.csv')
test=pd.read_csv('/content/gdrive/My Drive/CptS315Project/test.csv')
# view a small selection
print(train.head(10))

#Global parameters
exclude_stop_words = True
stopWords = stopwords.words('english')

# define functions to clean the text
def cleanWords(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def removeStopWords(text): 
   tokenizer = TreebankWordTokenizer()
   comment_tokens = tokenizer.tokenize(text)

   newText = [word for word in comment_tokens if word not in stopWords]
   return newText
 
def process(comment):
   """Process  function.
   Input:
      comment: a string containing a comment
   Output:
      comments_clean: a list of words containing the processed comment
   """
   stemmer = PorterStemmer()
   stopwords_english = stopwords.words('english')
   # remove stock market tickers like $GE
   comment = re.sub(r'\$\w*', '', comment)
   # remove old style text "RT"
   comment = re.sub(r'^RT[\s]+', '', comment)
   # remove hyperlinks
   comment = re.sub(r'https?:\/\/.*[\r\n]*', '', comment)
   # remove hashtags
   # only removing the hash # sign from the word
   comment = re.sub(r'#', '', comment)
   # tokenize comments
   tokenizer = TreebankWordTokenizer()
   comment_tokens = tokenizer.tokenize(comment)

   comments_clean = []
   for word in comment_tokens:
      if (word not in stopwords_english and  # remove stopwords
               word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            # possibly only add if the word is not ''
            comments_clean.append(stem_word)

   return comments_clean

# take out all puntcuation
train['comment_text'] = train['comment_text'].apply(lambda x: cleanWords(x))
print(train.head(10))
test['comment_text'] = test['comment_text'].apply(lambda x: cleanWords(x))

# clean stop words from train and test
train['comment_text'] = train['comment_text'].apply(lambda x: removeStopWords(x))
print(train.head(10))
test['comment_text'] = test['comment_text'].apply(lambda x: removeStopWords(x))
print(test.head(10))

# copied from kaggle
cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[cols].values

train_df = train['comment_text']
test_df = test['comment_text']

max_features = 22000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(train_df))

tokenized_train = tokenizer.texts_to_sequences(train_df)
tokenized_test = tokenizer.texts_to_sequences(test_df)

maxlen = 200
X_train = pad_sequences(tokenized_train, maxlen = maxlen)
X_test = pad_sequences(tokenized_test, maxlen = maxlen)

embed_size = 128
maxlen = 200
max_features = 22000

inp = Input(shape = (maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# print summary of model
print(model.summary())

batch_size = 64
epochs = 2
print(model.fit(X_train, targets, batch_size=batch_size, epochs=epochs, validation_split=0.1))


""" 
nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("       : train : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"   :",round(nrow_test*100/sum)) """

# sort vocab alphabetically
#vocab.sort()