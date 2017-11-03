import pandas as pd
import numpy as np
import json
import ast
import datetime
import time
import random
import re
from collections import defaultdict
import json


def load_data_from_jsons(filepath):
    """
    The steps necessary to take the file of a list of json objects and compile it
    into a pandas DataFrame
    INPUTS:
    filepath (string) - file path or name of file (made up of list of json
                        objects)
    OUTPUTS:
    df (pandas dataframe) - dataframe of the objects in filepath
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


def simplify(df, cols=None):
    """
    To reduce to number of columns to those I need to look at more closely (for EDA)
    """
    if not cols:
        cols = ['author_flair_css_class', 'author_flair_text', 'controversiality', 'distinguished', 'downs', 'edited', 'gilded', 'score', 'score_hidden', 'ups']
    return df.drop(cols, axis=1)


def prep_dates(infile, outfile='dates_prepped.csv', out=True):
    """
    Create a file and or dataframe which will be used to search the comment data
    set files (which are broken up into months, and named RC_yyyy-mm)
    INPUTS:
    infile - csv file which contains dates in epoch seconds and urls
             (in that order)
    out - boolean value, determines whether to write a csv of the results or not
    outfile - name of the csv file to write the data to, if out is set to True
    OUTPUTS:
    df - full dataframe, with columns of formatted date
    outfile - if out is set to True, returns csv with the columns file_name
             (date formatted to RC_yyyy-mm) and link_id (thread identifier
             fullname)
    """
    # read file into database
    df = pd.read_csv(infile, header=None, names=['epoch_date', 'url'])
    # remove nulls
    df.dropna(axis=0, inplace=True)
    # append column with datetimes
    df['date'] = pd.to_datetime(df['epoch_date'], unit='s')
    # make column for data-selection
    yr = df['date'].dt.year.apply(lambda x: str(int(x)))
    mo = df['date'].dt.month.apply(lambda x: str(int(x)).zfill(2))
    df['yr-month'] = 'RC_' + yr + '-' + mo
    # make column for link_ids
    df['link_id'] = df['url'].apply(lambda x: "t3_"+re.findall("comments/(.*?)/", x)[0])
    if out:
        out_df = df[['yr-month', 'link_id']].groupby('yr-month')
        df.to_csv(outfile, columns=['yr-month', 'link_id'], index=False)
    else:
        return df



def show_usefulness(df,useless=True):
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


def small_test_prep(ann_df, scraped_df, out=False):
    # remove nulls in scraped
    scraped_df.dropna(axis=0, inplace=True)
    # append columns
    df = pd.merge(ann_df, scraped_df, how='outer', on='url')
    df.drop(labels=['is_self_post'], axis=1, inplace=True)
    return df


def test(annotations, comments):
    # create set of annotation ids
    ann_ids = set()
    A = defaultdict(list)
    for thread in test_df['posts']:
        for item in thread:
            ann_ids.add(item['id'])
            A[item['id']] = item['annotations']
    # create set of comment names (same format as annotation ids)
    comment_set = set(comments['name'])
    #find intersection of sets
    inter = ann_ids.intersection(comment_set)
    # Identify parts we want to keep going forward:
    # take body from comments for intersecting names
    relevant_comments = comments[comments['name'].isin(inter)].set_index('name')
    # take relevant annotations
    D = {}
    for i in inter:
        if i in A:
            D[i] = A.get(i)
    # merge together
    ann_df = pd.DataFrame.from_dict(D,orient='index')
    ann_df.rename(columns={0:'ann_1', 1:'ann_2', 2:'ann_3'}, inplace=True)
    final_df = relevant_comments.join(ann_df)

    return final_df



if __name__ == '__main__':
    # make normalized_list_of_dates
    scraped_dates = 'data/list_of_dates.csv'

    # make a file with which to search the table
    df = prep_dates(scraped_dates, outfile='data/search_table.csv', out=False)

    #for visualization
    orig_json_file = 'data/coarse_discourse_dataset.json'
    coarse_df = load_data_from_jsons(orig_json_file)

    #load some data
    later_reddit_data = 'data/RC_2008-11'
    j = json.loads(later_reddit_data)

    later_df = load_data_from_jsons(later_reddit_data)
    useless = show_usefulness(df)
    simp_later = simplify(later_df)

    # smush together scraped dates, link_ids with annotation data
    test_df = small_test_prep(coarse_df, df)
