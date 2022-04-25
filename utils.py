import numpy as np
import pandas as pd 


import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
import re 

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from contractions import contractions_dict

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.model_selection import cross_val_score




def remove_characters_before_tokenization(text):
    text = text.strip()
    return re.sub(r'[^a-zA-Z0-9\' ]', r'', text)

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def expand_contra(sentence, contractions_dict):
    contras = re.findall(r'\w+\'\w+', sentence)
    for i in contras:
        expanded_contraction = contractions_dict.get(i)\
                               if contractions_dict.get(i)\
                               else contractions_dict.get(i.lower())
        if expanded_contraction:
            sentence = re.sub(i, expanded_contraction, sentence)
    return sentence

def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for index, text in enumerate(corpus):
        try:
            text = expand_contra(text, contractions_dict)
        except:
            print(index)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
    return normalized_corpus

def feat_extract(data,ngram_range):
    vectorizer = CountVectorizer(min_df=1,ngram_range=ngram_range)
    feature = vectorizer.fit_transform(data)
    return(vectorizer,feature)

def tfidf_transformer(matrix):
    transform = TfidfTransformer(norm='l2',smooth_idf=True,use_idf=True)
    tfidf_matrix = transform.fit_transform(matrix)
    
    return(transform, tfidf_matrix)

def metrics(clf_lst, X_test, y_test):
    
    metrics = []
    for clf in clf_lst:
        metrics_lst = []
        y_pred = clf.predict(X_test)
        metrics_lst.append(accuracy_score(y_true=y_test,y_pred=y_pred))
        metrics_lst.append(f1_score(y_true=y_test,y_pred=y_pred,average='weighted'))
        metrics_lst.append(recall_score(y_true=y_test,y_pred=y_pred,average='weighted'))
        metrics_lst.append(precision_score(y_true=y_test,y_pred=y_pred,average='weighted'))
        
        metrics.append(metrics_lst)
    return metrics

