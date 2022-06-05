import seaborn as sns
import matplotlib.pylab as plt


import numpy as np
import pandas as pd

import scipy
from scipy import sparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


import utils

import dataframe_image as dfi

import requests

train=pd.read_csv('train.csv')
test=pd.read_csv('test_with_solutions.csv')
anti_vac = pd.read_csv('antiva_dataset.csv')



train_comment = train['Comment']
test_comment = test['Comment']
train_label = train['Insult']
test_label = test['Insult']

a = anti_vac[anti_vac.nlikes < 200]
data_tweet = a['tweet']



def data_clean(Data_to_clean):
    
    Data_to_clean1 = [utils.remove_characters_before_tokenization(i) for i in Data_to_clean]
    normalized_data = utils.normalize_corpus(corpus=Data_to_clean1,tokenize=False)
    return normalized_data
    
    train_corpus = normalized_data[:3947]
    test_corpus = normalized_data[3947:]
    
    train_vec,train_feat = utils.feat_extract(data=train_corpus,ngram_range=(1,3))
    train_features = train_feat.todense()
    test_features = train_vec.transform(test_corpus).todense()
    
    train_transform , train_matrix = utils.tfidf_transformer(train_features)
    train_final_feature = train_matrix.todense()
    test_final_feature = train_transform.transform(test_features).todense()
    
    X_training,X_testing=sparse.csr_matrix(train_final_feature),sparse.csr_matrix(test_final_feature)


def ml_implement(X_train, y_train, X_test, y_test):
    
    classifiers_dict = []
    
    NB = MultinomialNB()
    NB.fit(X=X_train,y=y_train)
    classifiers_dict['NB'] = NB
    
    SGD = SGDClassifier()
    SGD.fit(X=X_train,y=y_train)
    classifiers_dict['SGD'] = SGD
    
    LogReg = LogisticRegression()
    LogReg.fit(X=X_train,y=y_train)
    classifiers_dict['LR'] = LogReg
    
    GB = GradientBoostingClassifier()
    GB.fit(X=X_train,y=y_train)
    classifiers_dict['GB'] = GB
    
    RF = RandomForestClassifier()
    RF.fit(X=X_train,y=y_train)
    classifiers_dict['RF'] = RF
    
    metrics = utils.metrics(classifiers_dict.values(), X_test, y_test)  
    
    df = pd.DataFrame(metrics, 
             columns=['Accuracy', 'F1_score', 'recall_score', 'precision_score'], 
             index = ['Naive Bayes', "SGD", "Logistic Regression", 'GradientBoosting', 'RandomForest'])
    
    dfi.export(df, 'clf_metrics.png')
    
    df.T.plot(kind='bar', figsize = (10, 10), )
    plt.xticks(rotation=360)
    plt.legend(loc='lower right')
    plt.savefig('clf_metrics.png')
    
    return classifiers_dict

def apply_clf(clf, tweets_final, dataset):
    sns.scatterplot(clf.predict_proba(tweets_final)[:, 1], dataset['nlikes'])

if __name__ == "__main__":
    
    Data_to_clean = pd.concat([train_comment,test_comment],axis=0)
    normalized_data = data_clean(Data_to_clean)
    train_corpus = normalized_data[:3947]
    test_corpus = normalized_data[3947:]
    
    train_vec,train_feat = utils.feat_extract(data=train_corpus,ngram_range=(1,3))
    train_features = train_feat.todense()
    test_features = train_vec.transform(test_corpus).todense()
    
    train_transform , train_matrix = utils.tfidf_transformer(train_features)
    train_final_feature = train_matrix.todense()
    test_final_feature = train_transform.transform(test_features).todense()
    
    X_training,X_testing=sparse.csr_matrix(train_final_feature),sparse.csr_matrix(test_final_feature)
    
    clf_dict = ml_implement(X_train, train_label, X_test, test_label)

    data_tweet1 = data_clean(data_tweet)
    data_features = train_vec.transform(data_tweet1).todense()
    data_final_feature = train_transform.transform(data_features).todense()
    final_data = sparse.csr_matrix(data_final_feature)
    
    
    clf = clf_dict['RF']
    
    data_relation = pd.DataFrame()
    
    data_relation['tweet'] = a['tweet']
    data_relation['prob'] = RF.predict_proba(final_data)[:, 1]
    data_relation['nlikes'] = a['nlikes']
    data_relation['nreplies'] = a['nreplies']
    data_relation['nreplies'] = a['nreplies']
    
    
    sns.scatterplot(RF.predict_proba(final_data)[:, 1], a['nlikes'])
    plt.xlabel('Probability of being politically incorrect');
    plt.title('The probablity of being politically incorrect vs. the number of likes');
    plt.savefig('ralation_nlikes.png')
    
    sns.scatterplot(RF.predict_proba(final_data)[:, 1], a['nreplies'])
    plt.xlabel('Probability of being politically incorrect');
    plt.title('The probablity of being politically incorrect vs. the number of replies');
    plt.savefig('ralation_nreplies.png')
    
    sns.scatterplot(RF.predict_proba(final_data)[:, 1], a['nretweets'])
    plt.xlabel('Probability of being politically incorrect');
    plt.title('The probablity of being politically incorrect vs. the number of retweets');
    plt.savefig('ralation_nretweets.png')
    
    
    
    # get the number of followers of each username
    count_list = []
    for username in a['username']:
        url = 'https://cdn.syndication.twimg.com/widgets/followbutton/info.json?screen_names='+username
        response  = requests.get(url)
        #print(response.json()[0]['followers_count'])
        #print(type(response.json()[0]['followers_count']))
        if response.json():
            count_list.append(response.json()[0]['followers_count'])
        else:
            count_list.append(None)
    
    data_relation['followers_count'] = count_list
    mean_value=df_relation['followers_count'].mean()
    df_relation['followers_count'].fillna(value = mean_value, inplace=True)
    d = df_relation[df_relation['followers_count']<3000]
    
    sns.lmplot(x='prob', y='followers_count', data=d, scatter_kws={"alpha":0.3, "s":5})
    plt.xlabel('Probability of being politically incorrect');
    plt.title('The probablity of being politically incorrect vs. the number of followers')
    plt.savefig('ralation_followers.png')
    
    sns.lmplot(x='followers_count', y='prob', data=d, scatter_kws={"alpha":0.2, "s":4})
    plt.ylabel('Probability of being politically incorrect');
    plt.title('The number of followers vs. the probablity of being politically incorrect')
    plt.savefig('followers_prob.png')
    
    sns.scatterplot(d['followers_count'], d.nlikes)
    plt.title('The number of followers vs. the number of likes');
    plt.savefig('followers_nlikes.png')
    