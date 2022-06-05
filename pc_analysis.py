import utils

from wordcloud import WordCloud, STOPWORDS
import string

import pandas as pd
import numpy as np
import word_freq

def data_clean(df):
    data = [utils.remove_characters_before_tokenization(i) for i in df]
    data1= utils.normalize_corpus(corpus=data,tokenize=False)
    
    text = " ".join(tweet for tweet in data1)
    return text

def plot_wordcloud(text):
    stopwords = set(STOPWORDS)
    stopwords.update(list(string.punctuation) + ["https", "people", 'think', 'will', 's', 'others', "one", "politically correct", "politically", "correct", "political correctness", "political", "correctness", "sensitive", 'covid','covid-19', 'covid19', "vaccines", 'vaxxer', 'vaxxers', 't', 'co', 'pandemic', 'anti-vaccine', 'amp'] + ['considerate', 'diplomatic', 'gender free', 'inclusive', 'inoffensive', 'multicultural', 'multiculturally sensitive', 'politic', 'respectful', 'sensitive', 'sensitive to others', 'bias free', 'liberal', 'nondiscriminatory', 'nonracist', 'nonsexist', 'unbiased', 'political correctness', 'politically correct'])



    wordcloud = WordCloud(stopwords=stopwords, background_color="white", min_word_length=4, collocation_threshold=4).generate_from_text(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('pc_wordcloud.png')
    
if __name__ == '__main__':
    df_pc= pd.read_csv('pc_dataset.csv', index_col=0)
    text = data_clean(df_pc['tweet'])
    
    word_freq.plot_top_adj(df1['tweet'], "PC tweets", n = 20)
    plt.savefig('top_adj')
    
    word_freq.plot_top_noun(df1['tweet'], "PC tweet", n = 20)
    plt.savefig('top_noun')
    
    word_freq.plot_top_lemmas(df1['tweet'], "tweet", n = 20)
    plt.savefig('top_lemma')
    
    plot_wordcloud(text)

