import pandas as pd
import datetime
import time
import ast
import re
from collections import defaultdict
import time


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


def prep_dates(infile):
    """
    Create a dataframe which will be used to search the comment data
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
    # make column for link_ids
    df['link_id'] = df['url'].apply(lambda x: "t3_"+re.findall("comments/(.*?)/", x)[0])

    return df


def prep_annotations(annotations, dates):
    """
    Converts post-based annotation table to comment-based table. First it merges
    dates with the annotation table, then unpacks the posts column into its own
    dataframe, then joins the two before cleaning up any nulls that might result.
    INPUTS:
    annotations (pandas df) - one row for every thread annotated
    dates (pandas df) - the date created for every thread in the annotations df
    OUTPUTS:
    combined (pandas df) - one row for every comment annotated
    """
    # merge the dataframes
    df = pd.merge(annotations, dates, how='outer', on='url')
    # unpack the posts column (which is annotations from all posts in a thread
    # join it to annotations df to make a comment specific dataframe
    r = []
    for i, post in enumerate(df['posts']):
        for comment in post:
            comment['ann_key'] = i
            r.append(comment)
    posts_df = pd.DataFrame(r).set_index('ann_key')
    combined = posts_df.join(df)
    combined.reset_index(drop=True, inplace=True)
    combined.drop(['posts', 'epoch_date', 'url', 'is_self_post'], axis=1, inplace=True)
    #fix resulting nulls
    fills = {"post_depth":0, "in_reply_to":'None', 'is_first_post':False, 'majority_link':'None', 'majority_type':'None'}
    for col in fills:
        combined[col].fillna(value=fills[col],  inplace=True)
    #drop threads from banned or private subreddits (can't get comment text)
    combined.dropna(inplace=True)

    return combined


def search_prep(ann_df, scraped_df):
    """ No longer in use"""
    # append columns
    df = pd.merge(ann_df, scraped_df, how='outer', on='url')
    df.drop(labels=['is_self_post'], axis=1, inplace=True)
    return df

def find_annotated_comments(annotations, comments):
    # create set of annotation ids
    ann_ids = set()
    A = defaultdict(list)
    for thread in annotations['posts']:
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


def search_files(annotations, file_list):
    searched = {}
    results = []
    for f in file_list:
        comment_df = load_data_from_jsons(f)
        df = find_annotated_comments(annotations, comment_df)
        searched[f] = df
        results.append(df)

    final_results = pd.concat(results)
    return final_results


if __name__ == '__main__':
    #use datetime for debugging purposes:
    START1 = time.time()
    #create and prep date dataframe (date, url, yr-month key, link_id, by link_id)
    date_file = "data/list_of_dates.csv"
    dates = prep_dates(date_file)
    T1 = time.time()
    print("dates prepped, took {}".format(T1-START1))

    # load annotations dataframe (post, url, subreddit, et alia, by url)
    START2 = time.time()
    annotation_file = 'data/coarse_discourse_dataset.json'
    annotations = load_data_from_jsons(annotation_file)
    T2 = time.time()
    print("base annotations loaded, took {}".format(T2-START2))

    # merge annotations with its posts section
    START3 = time.time()
    search_table = prep_annotations(annotations, dates)
    T3 = time.time()
    print("search table constructed, took {}".format(T3-START3))

    """
    #small test
    START4 = time.time()
    comment_file = 'data/RC_2008-11'
    comment_df = load_data_from_jsons(comment_file)
    T4 = time.time()
    START5 = time.time()
    print("smaller reddit file loaded, took {}".format(T4-START4))
    test_results = find_annotated_comments(annotations, comment_df)
    T5 = time.time()
    print("1 file test complete, took {}".format(T5-START5))

    #larger run
    START6 = time.time()
    file_list = ["data/RC_2008-11", "data/RC_2008-08"]
    test_files_results = search_files(annotations, file_list)
    T6 = time.time()
    print("2 file test complete, took {}".format(T6-START6))
    """
