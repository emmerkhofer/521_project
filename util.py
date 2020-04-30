#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:32:13 2020

@author: yoyozhang
"""
import os
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from string import digits
import pandas as pd
from sklearn.model_selection import cross_val_score
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
import numpy as np
import datetime
from keras import models, layers
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")




def get_data(path, first_level, second_level):
    files= os.listdir(path + '/data/' + first_level + '/' + second_level) 
    s = []
    for file in files:
         if not os.path.isdir(path + '/data/' + first_level + '/' + second_level + '/' + file): 
             with open(path + '/data/' + first_level + '/' + second_level + '/' + file, encoding="utf8", errors='ignore') as f: 
              s.extend(f.readlines())
          
              
    return s

def get_dataframe(data, sentiment, label):
    df = pd.DataFrame(columns = ['review', 'sentiment', 'label'])
    df['review'] = data
    if sentiment == 'negative':
        df['sentiment'] = 0
    else:
        df['sentiment'] = 1
        
    if label == 'deceptive':
        df['label'] = 1
    else:
        df['label'] = 0
        
    return df
    
    

    

def remove_punctuation(text):
    
    punctuation = '!,;:?"\'.$@'
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()


def preprocess(reviews):
    #remove punctuation
    reviews = remove_punctuation(reviews)
    
    #review non-letter characters
    remove_digits = str.maketrans('', '', digits)
    reviews = reviews.translate(remove_digits)
    
    #lower all the letters
    reviews = reviews.lower()
    
    #word stemmer
    reviews = reviews.split()
    ps=PorterStemmer()
    reviews = [ps.stem(r) for r in reviews]
    
    #remove stopwords
    stoplist = list(stopwords.words('english'))
    reviews = ' '.join([i for i in reviews if i not in stoplist])
    
    return reviews
    

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

def N_gram_tokenizer(reviews):
    tokenizer = []
    tokenizer.append([extract_ngrams(i, 1) for i in reviews])
    tokenizer.append([extract_ngrams(i, 2) for i in reviews])
    tokenizer.append([extract_ngrams(i, 3) for i in reviews])
    tokenizer.append([extract_ngrams(i, 4) for i in reviews])

    
    return tokenizer

def TF(corpus, ngram, max_feature):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram, ngram),max_features = max_feature) 
    X = vectorizer.fit_transform(corpus)
    count = X.toarray()
    row = np.shape(count)[0]
    col = np.shape(count)[1]
    
    return count, row, col


def TFIDF(corpus, ngram, max_feature):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(ngram, ngram),max_features = max_feature)
    X = vectorizer.fit_transform(corpus)
    count = X.toarray()
    row = np.shape(count)[0]
    col = np.shape(count)[1]
    
    return count, row, col
    
def wordcloud(df, group):
    text = " ".join(review for review in df.review)
    print ("There are {} words in the combination of all review.".format(len(text)))
    
    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["room", "hotel", "desk", "Chicago", "stay", "day"])
    
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, collocations=False, max_words=100,
                          background_color="white").generate(text)
    
    # Display the generated image:
    # the matplotlib way:
    plt.figure( figsize=(10,8) )
    plt.title("Word Cloud for " + group + ' reviews', fontsize=18)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    
    
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_top_bigram(corpus, color, group):
    common_words = get_top_n_bigram(corpus, 20)
    df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
        kind='bar', figsize=(10, 8), color = color, fontsize = 11,
        title= 'Top 20 bigrams in reviews ' + group)
    plt.show()
    


#input the layer and neurons in each layer, e.g(8,8,1) reprensents there is 3 layers, each layer has neuron number of 8,8,1 respectively
def create_CNN(feature_number, network_structure):
    clf = models.Sequential()
    clf.add(layers.Dense(network_structure[0], activation='relu', input_dim = feature_number))
    for layer in network_structure[1:-1]:
        clf.add(layers.Dense(layer,activation = 'relu'))
    
    clf.add(layers.Dense(network_structure[-1], activation='sigmoid'))
    clf.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    
    return clf
  


def run_model(model_name, review, X, y,CNN_param_list = None):
    starttime = datetime.datetime.now()
    names = globals()
    print(model_name)
    for ngram in [1,2,3,4]:
        for feature in ['TF','TFIDF']:
            for max_feature in [1000, 5000, 10000]:
                print('the accuracy for ',ngram,'gram',feature,'with max feature number of',max_feature,'is')
                tf_feature, row, col = names[feature](review,ngram,max_feature)
                train = np.concatenate((X,tf_feature),axis = 1)
                if model_name == 'SVM':
                    clf = svm.SVC(kernel='linear', C=1)
                    scores = cross_val_score(clf, train, y, cv=5,scoring='accuracy')
                    print(scores.mean().round(4))
                    
                elif model_name == 'DT':
                    clf = DecisionTreeClassifier(random_state=0, max_depth=2)
                    scores = cross_val_score(clf, train, y, cv=5,scoring='accuracy')
                    print(scores.mean().round(4))
                    
                elif model_name == 'Xgboost':
                    clf = xgb.XGBClassifier()
                    scores = cross_val_score(clf, train, y, cv=5,scoring='accuracy')
                    print(scores.mean().round(4))
                    
                    
                elif model_name == 'RandomForest':
                    clf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 42)
                    scores = cross_val_score(clf, train, y, cv=5,scoring='accuracy')
                    print(scores.mean().round(4))
                    
                elif model_name == 'LR':
                    clf = LogisticRegression(random_state=0)
                    scores = cross_val_score(clf, train, y, cv=5,scoring='accuracy')
                    print(scores.mean().round(4))
                    
                elif model_name == 'CNN':
                    y = np.array(y)
                    y.resize(1604,1)
                    for param in CNN_param_list:
                        print('for CNN with param ', param)
                        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
                        scores = []
                        clf = create_CNN(col + 2, param)
                        for train_idx, test_idx in kfold.split(train, y):
                            clf.fit(train[train_idx], y[train_idx], epochs=10, batch_size=32, verbose=0)
                            cvscore = clf.evaluate(train[test_idx], y[test_idx], verbose=0)
                            scores.append(cvscore[1])
                        
                        scores = np.array(scores)
                        
                        print(scores.mean().round(4))
                        

                
    endtime = datetime.datetime.now()
    print('the running time is', (endtime - starttime).seconds)
    

    
