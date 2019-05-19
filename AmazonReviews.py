# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:34:44 2019

@author: andre
"""
import os
import json
import pandas as pd
import numpy as np
from re import sub, compile
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn import metrics, model_selection
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
import nltk
import gensim
from gensim.models import LdaModel, LsiModel
import warnings; warnings.simplefilter('ignore')



df_data = pd.read_csv("amazon_review_texts.csv", header = 0, 
                      names = ["pid", "helpful", "score", "text", "category"])

print(df_data.head())
print(df_data["score"].value_counts())
print(df_data["category"].value_counts())

categories = [
    'electronics',
    'software',
    'automotive',
    'watch',
]

stopwords = set(nltk.corpus.stopwords.words("english"))

import re

def before_token(documents):
    lower = map(str.lower, documents)
    punctuationless = list(map(lambda x: " ".join(re.findall('\\b\\w\\w+\\b', x)), lower))
    return list(map(lambda x:re.sub('\\b[0-9]+\\b', '', x), punctuationless))


stemmer = nltk.stem.PorterStemmer()
fdist = nltk.FreqDist()


def preprocess(doc):
    tokens = []
    for token in doc.split():
        if token not in stopwords:
            tokens.append(stemmer.stem(token))
    return tokens


processed = list(map(preprocess, before_token(df_data['text'])))
print(processed[1])

fdist = nltk.FreqDist([token for doc in processed for token in doc])
fdist.tabulate(10)


processed_doc = list(map(" ".join, processed))
vectorizer = TfidfVectorizer(max_df=0.8, norm = 'l2', stop_words='english')
X = vectorizer.fit_transform(processed_doc)

print("n_samples: %d, n_features: %d" % X.shape)


km = KMeans(n_clusters = len(categories), max_iter=100, random_state=54321)
km.fit(X)


order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(len(categories)):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])


corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id2word = dict((v,k) for k,v in vectorizer.vocabulary_.items())

lda4 = LdaModel(corpus, num_topics = 4, id2word=id2word, passes = 10)
print(lda4.print_topics())



from sklearn.linear_model import SGDClassifier
# 5-fold cross validation
skf = StratifiedKFold(n_splits=5)
fold = 0
f1 = []
for train_index, test_index in skf.split(df_data["text"], df_data["score"]):
#for train_index, test_index in skf:
    fold += 1
    print("Fold %d" % fold)
    # partition
    train_x, test_x = df_data["text"].iloc[train_index], df_data["text"].iloc[test_index]
    train_y, test_y = df_data["score"].iloc[train_index], df_data["score"].iloc[test_index]
    # vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df = 2,stop_words='english')
    X = vectorizer.fit_transform(train_x)
    print("Number of features: %d" % len(vectorizer.vocabulary_))
    X_test = vectorizer.transform(test_x)
    # train model
    clf = SGDClassifier(random_state=fold)
    clf.fit(X, train_y)
    # predict
    pred_y = clf.predict(X_test)
    # classification results
    for line in metrics.classification_report(test_y, pred_y).split("\n"):
        print(line)
    f1.append(metrics.f1_score(test_y, pred_y, average='weighted'))
print("Average F1: %.2f" % np.mean(f1))



df_data['satisfaction'] = df_data['score']
df_data['satisfaction'] = df_data['satisfaction'].replace([5,4,3,2,1],['1','1','0','0','0'])




skf = StratifiedKFold(n_splits=5)
fold = 0
f1 = []
for train_index, test_index in skf.split(df_data["text"], df_data["satisfaction"]):
#for train_index, test_index in skf:
    fold += 1
    print("Fold %d" % fold)
    # partition
    train_x, test_x = df_data["text"].iloc[train_index], df_data["text"].iloc[test_index]
    train_y, test_y = df_data["satisfaction"].iloc[train_index], df_data["satisfaction"].iloc[test_index]
    # vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df = 2,stop_words='english')
    X = vectorizer.fit_transform(train_x)
    print("Number of features: %d" % len(vectorizer.vocabulary_))
    X_test = vectorizer.transform(test_x)
    # train model
    clf = SGDClassifier(random_state=fold)
    clf.fit(X, train_y)
    # predict
    pred_y = clf.predict(X_test)
    # classification results
    for line in metrics.classification_report(test_y, pred_y).split("\n"):
        print(line)
    f1.append(metrics.f1_score(test_y, pred_y, average='weighted'))
print("Average F1: %.3f" % np.mean(f1))


