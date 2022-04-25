import twint

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd



anti_vax_keywords = ['anti vaccine', 'anti vaccination', '#antivax', '#antivaccine', '#antivaxxers', '#novaccine', '#novax']
pc_synonyms = ['considerate', 'diplomatic', 'gender free', 'inclusive', 'inoffensive', 'multicultural', 'multiculturally sensitive', 'politic', 'respectful', 'sensitive', 'sensitive to others', 'bias free', 'liberal', 'nondiscriminatory', 'nonracist', 'nonsexist', 'unbiased', 'political correctness', 'politically correct']

def fetch_by_keywords(keywords_lst, file_path):
    """Fetch tweets by keywords using Twint, and save them as a csv file.
    
    Input:
        keywords_lst(list): a list of keywords
        file_path(str): the path to save the csv file.
        

    """
    
    frame = []
    for keyword in keywords_lst:
        c=twint.Config()
        c.Search= keyword

        c.Pandas= True #Enable Pandas integration.

        c.Limit= 50000

        c.Hide_output = True
        twint.run.Search(c)

        df = twint.storage.panda.Tweets_df
        frame.append(df)
    result = pd.concat(frame)
    
    result[result.language == 'en'].to_csv(file_path)

if __name__ == "__main__":
    fetch_by_keywords(anti_vax_keywords, './antiva_dataset.csv')
    fetch_by_keywords(pc_synonyms, './pc_dataset.csv')
