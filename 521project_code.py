#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:28:01 2020

@author: yoyozhang
"""

import os
from util import run_model, create_CNN, get_data, preprocess, extract_ngrams, N_gram_tokenizer,TF, TFIDF, get_top_n_bigram,  wordcloud, get_dataframe, plot_top_bigram
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_validate
import warnings
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")





##################1. READ THE DATA ###################
#read seperately
path = os.getcwd()
neg_deceptive = get_data(path, 'negative','deceptive')
neg_truthful = get_data(path, 'negative', 'truthful')
pos_deceptive = get_data(path, 'positive','deceptive')
pos_truthful = get_data(path, 'positive', 'truthful')

#Constuct complete data
data = pd.DataFrame(columns = ['review', 'sentiment', 'label'])

neg_fake_df = get_dataframe(neg_deceptive, 'negative', 'deceptive')
neg_true_df = get_dataframe(neg_truthful, 'negative', 'truthful')
pos_fake_df = get_dataframe(pos_deceptive, 'positive', 'deceptive')
pos_true_df = get_dataframe(pos_truthful, 'positive', 'truthful')

data = pd.concat([neg_fake_df, neg_true_df, pos_fake_df, pos_true_df])
data = data.reset_index(drop = True)

###############2. PREPROCESS THE DATA###################
#2.1 - basic preprocess
data['review'] = [preprocess(i) for i in data['review']]

#2.2 Feature extraction
#2.2.1 TF or TFIDF
#it's constructed in util.py and directly used in part 3

#2.2.2 the sentiment of review
#this feature was constructed in part1

#2.2.3 the number words in the review
data['length'] = [len(i) for i in data['review']]

y = data['label']
X = np.array(data[['sentiment','length']])
review = data['review']


############3. exploratory analysis########################
##3.1 Word Cloud for top sigle words in reviews

wordcloud(neg_fake_df, 'negative (deceptive)')
wordcloud(neg_true_df, 'negative (true)')
wordcloud(pos_fake_df, 'positive (deceptive)')
wordcloud(pos_fake_df, 'positive (true)')

## 3.2. Bar chart for top bigrams 

plot_top_bigram(neg_fake_df['review'], 'deepskyblue', 'negative (deceptive)')
plot_top_bigram(neg_true_df['review'], 'lightblue', 'negative (true)')
plot_top_bigram(pos_fake_df['review'], 'pink', 'positive (deceptive)')
plot_top_bigram(pos_true_df['review'], 'salmon', 'positive (true)')


#######################4. Model Building##################################
#4.1 SVM
#features range from 3 dimention: gram number, feaure type(TF or TFIDF), the max number of feaures
run_model('SVM', review, X, y)
run_model('DT', review, X, y)
run_model('LR',review, X,y)
run_model('RandomForest', review, X, y)
run_model('Xgboost', review, X,y)

CNN_parameter_list = [(8,8,1),(8,1)]
run_model('CNN', review, X,y, CNN_parameter_list)

    