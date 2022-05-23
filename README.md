# replication-materials-zbchen0129
replication-materials-zbchen0129 created by GitHub Classroom
My research question is 1) how political correctness is discussed and portrayed on Twitter; 2) whether being politically correct/incorrect will affect the influence of an anti-vaccination tweet. 

All codes and data in this repository is written in Python 3.7 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```python
pip install -r requirements.txt
```

### Data Collection

Code for data collection part is here [data_collect.py](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/data_collect.py). You can run the following in  the terminal to get the anti-vaccine tweets data ([antiva_dataset.csv](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/antiva_dataset.csv)) and pc tweets data ([pc_dataset.csv](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/pc_dataset.csv)):

```python
run data_collect.py
```

Or use the `fetch_by_keywords` function defined in data_collect.py:

```python
import data_collect
data_collect.fetch_by_keywords(anti_vax_keywords, '/antiva_dataset.csv')
data_collect.fetch_by_keywords(pc_synonyms, '/pc_dataset.csv')
```

Besides, I also get a twitter dataset, which has been already labelled with whether being a hate speech, from [Kaggle](https://www.kaggle.com/competitions/detecting-insults-in-social-commentary/data). I downloaded training data ([train.csv](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/train.csv)) and testing data ([test_with_solutions.csv](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/test_with_solutions.csv)) to train the machine learning classification model.


### Methods and results

(1)

For my first question, I use word frequency analysis, including plotting top 20 adjectives, nouns and lemmas of the PC tweets and a wordcloud to analyze the tweets. Code for the first question is here [pc_analysis.py](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/pc_analysis.py). You can run the following in the terminal:

```python
import pc_analysis
import pandas as pd
```

And then use functions in pc_analysis.py:
```python
df_pc= pd.read_csv('pc_dataset.csv', index_col=0)
text = pc_analysis.data_clean(df_pc['tweet'])
pc_analysis.plot_wordcloud(text)
```

Or just run the whole file:
```python
run pc_analysis.py
```

After running it, you will get and save three plots and an image of the wordcloud. The three plots are shown below:

![Figure top nouns](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/top_noun.png)
![Figure top adjs](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/top_adj.png)
![Figure top lemmas](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/top_lemmas.png)


The image below is the wordcloud that shows the most occurring words that most characterize the discussions about political correctness. 

![Figure 1 pc wordcloud](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/pc_wordcloud.png)

This wordcloud shows that political correctness on Twitter is most linked to the topic of freedom, like free expression, gender issues and self identity. Besides, there are negative words like ‘hoax’ and ‘bias’, which to some extent reflects people's attitudes and opinions about PC on Twitter. Some words also identify PC from a political point of view in the top frequencies, like ‘democrats’. 


(2)

For the second question, I use machine learning classifier to predict the probability of a anti-vaccine tweet being politiclly incorrect, and then plot the relationship between the probablity of being politiclly correct and the number of likes, replies, retweets respectively. Code here [ml_clf.py](http://localhost:8888/edit/ml_clf.py)

The code is a bit long and complex. Therefore, I recommend you to directly run the file like the following:
```python
run ml_clf.py
```

You will get three plots that show the relationship between the probablity of being politiclly correct and the number of likes, replies, retweets respectively. Figures are below.

![Figure 2](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/ralation_nlikes.png)

![Figure 3](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/ralation_nreplies.png)

![Figure 4](https://github.com/macs30200-s22/replication-materials-zbchen0129/blob/main/ralation_nretweets.png)

From these three scatter plot, we can find that the higher probability, or say, the higher degree, of being political correctness, will lead the tweet to have more likes, replies and retweets in general. It also means that those tweets that have larger social influence (because more people are viewing, clicking like, replying and retweeting), typically will be more politically correct.

### Cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6485991.svg)](https://doi.org/10.5281/zenodo.6485991)
