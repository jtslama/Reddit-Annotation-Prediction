import pandas as pd
from initial_data_loading import load_discourse_data


def load_reddit_data(filepath):
    """
    experimental loading of reddit data for viewing
    outputs file from filepath as a pandas dataframe
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


def _simplify(df, cols=None):
    """
    To reduce to number of columns to those I need to look at more closely (for EDA)
    """
    if not cols:
        cols = ['author_flair_css_class', 'author_flair_text', 'controversiality', 'distinguished', 'downs', 'edited', 'gilded', 'score', 'score_hidden', 'ups']
    return df.drop(cols, axis=1)


def load_date_url_table(filepath):
    return pd.read_csv(filepath)



"""
Process:
I need to find the date ranges for which I need to download reddit data.
"""
