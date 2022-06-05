import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
from gensim import corpora, models
import seaborn as sns

from nltk.corpus import wordnet

# Define stop words + punctuation + study-specific stop-words
STOP = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ["https", "people", 'think', 'will', 's', 'others', "one", "politically correct", "politically", "correct", "political correctness", "political", "correctness", "sensitive"] + ['considerate', 'diplomatic', 'gender free', 'inclusive', 'inoffensive', 'multicultural', 'multiculturally sensitive', 'politic', 'respectful', 'sensitive', 'sensitive to others', 'bias free', 'liberal', 'nondiscriminatory', 'nonracist', 'nonsexist', 'unbiased', 'political correctness', 'politically correct']


def pos_tag(text):
    '''
    Tags each word in a string with its part-of-speech indicator, excluding stop-words
    '''
    # Tokenize words using nltk.word_tokenize, keeping only those tokens that do not appear in the stop words we defined
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in STOP and wordnet.synsets(i) and len(i) >= 4]

    # Label parts of speech automatically using NLTK
    pos_tagged = nltk.pos_tag(tokens)
    return pos_tagged

def plot_top_adj(series, data_description, n = 15):
    '''
    Plots the top `n` adjectives in a Pandas series of strings.
    '''
    # Apply part of Speech tagger that we wrote above to any Pandas series that pass into the function
    pos_tagged = series.apply(pos_tag)

    # Extend list so that it contains all words/parts of speech for all the captions
    pos_tagged_full = []
    for i in pos_tagged:
        pos_tagged_full.extend(i)

    # Create Frequency Distribution of different adjectives and plot the distribution
    fd = nltk.FreqDist(word for (word, tag) in pos_tagged_full if tag[:2] == 'JJ').most_common(n)
    #fd.plot(n, title='Top {} Adjectives in '.format(n) + data_description);
    
    all_fdist = pd.Series(dict(fd))
    
    fig, ax = plt.subplots(figsize=(8,8))
    palette = sns.color_palette("flare",n_colors=20)
    palette.reverse()

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(y=all_fdist.index, x=all_fdist.values, ax=ax, orient='h', palette=palette)
    for p in ax.patches:
        ax.annotate("%d" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),\
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    plt.title('Top {} adjectives in '.format(n) + data_description);
    #fd.plot(n, title='Top {} nouns in '.format(n) + data_description);
    plt.xlabel('Counts');
    plt.ylabel('Words');
    
    return

def plot_top_noun(series, data_description, n = 15):
    '''
    Plots the top `n` nouns in a Pandas series of strings.
    '''
    # Apply part of Speech tagger that we wrote above to any Pandas series that pass into the function
    pos_tagged = series.apply(pos_tag)

    # Extend list so that it contains all words/parts of speech for all the captions
    pos_tagged_full = []
    for i in pos_tagged:
        pos_tagged_full.extend(i)

    # Create Frequency Distribution of different adjectives and plot the distribution
    fd = nltk.FreqDist(word for (word, tag) in pos_tagged_full if tag[:2] == 'NN' and tag != 'NNP').most_common(n)
    all_fdist = pd.Series(dict(fd))
    
    fig, ax = plt.subplots(figsize=(8,8))
    palette = sns.color_palette("flare",n_colors=20)
    palette.reverse()

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(y=all_fdist.index, x=all_fdist.values, ax=ax, orient='h', palette=palette)
    for p in ax.patches:
        ax.annotate("%d" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),\
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    plt.title('Top {} nouns in '.format(n) + data_description);
    #fd.plot(n, title='Top {} nouns in '.format(n) + data_description);
    plt.xlabel('Counts');
    plt.ylabel('Words');
    
    return

def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well as a set of study-specific stop-words
    '''
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in STOP and wordnet.synsets(i) and len(i) >= 4]
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return lemmas

def plot_top_lemmas(series, data_description, n = 20):
    '''
    Plots the top `n` lemmas in a Pandas series of strings.
    '''
    lemmas = series.apply(get_lemmas)

    # Extend list so that it contains all words/parts of speech for all the captions
    lemmas_full = []
    for i in lemmas:
        lemmas_full.extend(i)

    fd = nltk.FreqDist(lemmas_full).most_common(n)
    all_fdist = pd.Series(dict(fd))
    
    fig, ax = plt.subplots(figsize=(8,8))
    palette = sns.color_palette("flare",n_colors=20)
    palette.reverse()

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(y=all_fdist.index, x=all_fdist.values, ax=ax, orient='h', palette=palette)
    for p in ax.patches:
        ax.annotate("%d" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),\
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    plt.title('Top {} lemmas in '.format(n) + data_description);
    plt.xlabel('Counts');
    plt.ylabel('Words');
    
    
    return

def plot_top_tfidf(series, data_description, n = 15):
    '''
    Plots the top `n` TF-IDF words in a Pandas series of strings.
    '''
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)

    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary([i for i in lemmas])

    # Convert dictionary into bag of words format: list of (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]

    # Calculate TFIDF based on bag of words counts for each token and return weights:
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals = 2)

    # highest TF-IDF values:
    top_n = pd.Series(tfidf_weights).nlargest(n)
    
    fig, ax = plt.subplots(figsize=(8,8))
    palette = sns.color_palette("flare",n_colors=20)
    palette.reverse()

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(y=top_n.index, x=top_n.values, ax=ax, orient='h', palette=palette)
    for p in ax.patches:
        ax.annotate("%d" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),\
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    plt.title('Top {} Lemmas (TFIDF) in '.format(n) + data_description);

    # Plot the top n weighted words:

    plt.xticks(rotation='vertical')
    plt.xlabel('Counts');
    plt.ylabel('Words');

    return
