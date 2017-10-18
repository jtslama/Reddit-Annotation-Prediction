import pandas as pd
import json
import ast
import datetime
import time
import random
import matplotlib.pyplot as plt


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


def _simplify(df, cols=None):
    """
    To reduce to number of columns to those I need to look at more closely (for EDA)
    """
    if not cols:
        cols = ['author_flair_css_class', 'author_flair_text', 'controversiality', 'distinguished', 'downs', 'edited', 'gilded', 'score', 'score_hidden', 'ups']
    return df.drop(cols, axis=1)


def _normalize_the_time_flow(filename):
    df = pd.read_csv(filename, header=None, names=['epoch_date', 'url'])
    df['date'] = pd.to_datetime(df['epoch_date'], unit='s')
    return df


def _show_usefulness(df,useless=True):
    m, u = {}, {}
    for yr in xrange(2007,2016):
        for mnth in xrange(1,13):
            s = len( df[(df['date'].dt.month==mnth) & (df['date'].dt.year==yr)] )
            if s > 0:
                m["{}-{}".format(yr,mnth)] = s
            if u.get(s):
                u[s].append( "{}-{}".format(yr,mnth) )
            else:
                u[s] = ["{}-{}".format(yr,mnth)]

    if useless:
        return u.get(0, None)
    return m.keys()



if __name__ == '__main__':
    #for testing
    scraped_dates = 'data/list_of_dates.csv'
    df = _normalize_the_time_flow(scraped_dates)
    df.url[0]
    unnecessary = _show_usefulness(df)
    nec = _show_usefulness(df, useless=False)


    #for visualization
    orig_json_file = 'data/coarse_discourse_dataset.json'
    coarse_df = load_discourse_data(orig_json_file)

    for subreddit in coarse_df.subreddit.unique():
        print subreddit
    coarse_df[coarse_df.subreddit == 'circlejerk']['posts'].iloc[0]
    len(coarse_df)
    coarse_df.url[1982]
    list(coarse_df.url[:10])
    len(coarse_df)

    # looking at a small sample of reddit data
    %pwd
    sample_reddit_data = 'data/RC_2008-06'
    comment_df = load_reddit_data(sample_reddit_data)
    comment_df[comment_df.link_id == 't3_7ajhj']
    comment_df.head()
    comment_df.distinguished.unique()

    later_reddit_data = 'data/RC_2008-11'
    later_df = load_reddit_data(sample_reddit_data)

    # there's a ton. Let's get rid of some columns I don't need
    early = _simplify(comment_df)
    later = _simplify(later_df)
    a = early[early['link_id']=='t3_6liww']
    b = later[later['link_id']=='t3_6liww']
    edits = []
    for l_id in early.link_id:
        a = len( early[early['link_id']==l_id] )
        b = len( later[later['link_id']==l_id] )
        if b != a:
            print "{} grew by {}".format(l_id, b-a)
            edits.append([l_id, (a,b)])



    len(a), len(b)
    less_reddit.head()
    less_reddit[less_reddit['link_id'] == 't3_7ajhj']
    less_reddit.columns
    len(less_reddit.id.unique())
    less_reddit.link_id[1]
    less_reddit.retrieved_on.max()
    """
