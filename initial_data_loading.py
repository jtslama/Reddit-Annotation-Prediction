import pandas as pd
import json
import ast
import requests
from bs4 import BeautifulSoup
import time
import random


def load_discourse_data(filepath):
    """
    Since I keep doing it, and its a pain every time:
    The steps necessary to take the file of a list of json objects and compile it
    into a pandas DataFrame
    """
    # read file into a list
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # remove newline chars, change true (invalid) to True (bool)
    rows = [lin.rstrip().replace('true', 'True') for lin in lines]
    rows_fixed = [r.replace('true', 'True') for r in rows]
    # now each string can be made to a dictionary
    dictified = [ast.literal_eval(rf) for rf in rows_fixed]
    # then loaded into a pandas DataFrame
    df = pd.DataFrame(dictified)
    return df




def load_reddit_data(filepath):
    """
    Since I keep doing it, and its a pain every time:
    The steps necessary to take the file of a list of json objects and compile it
    into a pandas DataFrame
    """
    filepath = 'data/RC_2008-11'
    # read file into a list
    with open(filepath, 'r') as f:
        rows = f.readlines()
    # remove newline chars, change true (invalid) to True (bool), false to False, null to None
    switch = {'true': 'True',
            'false': 'False',
            'null': 'None'}
    for item in switch:
        rows = [r.replace(item, switch[item]) for r in rows]
    rows = [r.rstrip() for r in rows]
    # now each string can be made to a dictionary
    dictified = [ast.literal_eval(rf) for rf in rows]
    # then loaded into a pandas DataFrame
    df = pd.DataFrame(dictified)
    return df


def construct_urls(df):
    starter_url = 'www.reddit.com/r/'
    urls = []
    for num in df['link_id']:
        # num consists of id_type (eg t2) + _ + id, we only want id
        url = starter_url + num[3:]
        urls.append(url)
    return urls

"""
def _find_link_ids(df):
    url = 'https://www.reddit.com/r/100movies365days/comments/1bx6qw/dtx120_87_nashville/'

    for url in df['url']
    url_components = url.split('/')
    link_id = 't3_'+ url_components[ url_components.index('comments')+1]
"""




def _simplify(df, cols=None):
    """
    To reduce to number of columns to those I need to look at more closely (for EDA)
    """
    if not cols:
        cols = ['author_flair_css_class', 'author_flair_text', 'controversiality', 'distinguished', 'downs', 'edited', 'gilded', 'score', 'score_hidden', 'ups']
    return df.drop(cols, axis=1)




if __name__ == '__main__':
    orig_json_file = 'data/coarse_discourse_dataset.json'
    coarse_df = load_discourse_data(orig_json_file)
    coarse_df.head()
    list(coarse_df.url[:10])
    len(coarse_df)

    # looking at a small sample of reddit data
    sample_reddit_data = 'data/RC_2008-11'
    comment_df = load_reddit_data(sample_reddit_data)
    comment_df.columns
    comment_df.head()
    comment_df.distinguished.unique()
    # there's a ton. Let's get rid of some columns I don't need
    less_reddit = _simplify(comment_df)
    less_reddit.head()
    less_reddit.columns
    less_reddit.ix[1]
    len(less_reddit.id.unique())
    less_reddit.link_id[1]
    less_reddit.retrieved_on.max()
