import pandas as pd
import datetime
import time
import sys
import json
import ast
from reduce_comments import CommentFinder
from sklearn.model_selection import train_test_split



def base_label_splitting(df):
    df['annotations'] = df['annotations'].map(lambda x: ast.literal_eval(x))
    df['ann_len'] = df['annotations'].map(lambda x: len(x))

    for i in xrange(1,4):
        col = "ann_{}".format(i)
        df[col] = None

    df['ann_1'] = df['annotations'].map(lambda x: x[0]['main_type'])
    df['ann_2'] = df.apply(lambda row: _ann_split_2(row), axis=1)
    df['ann_3'] = df.apply(lambda row: _ann_split_2(row), axis=1)

    return df


def _ann_split_2(row):
    if row['ann_len'] == 3:
        return row['annotations'][1]['main_type']
    elif row['ann_len']== 2:
        return row['annotations'][1]['main_type']
    elif row['ann_len'] == 1:
        return None


def _ann_split_3(row):
    if row['ann_len'] == 3:
        return row['annotations'][2]['main_type']
    elif row['ann_len']== 2:
        return None
    elif row['ann_len'] == 1:
        return None


# def less_extra_colsdf, cols=None):
#     df.drop(['ann_len'], axis=1, inplace=True)
#     df.drop(cols, axis=1, inplace=True)
#     return df


def clean(df, extra_cols=None):
    df = base_label_splitting(df)
    df.drop(['ann_len'], axis=1, inplace=True)
    if extra_cols:
        df.drop(extra_cols, axis=1, inplace=True)
    return df




if __name__ == '__main__':
    # load file
    left, inner = '/Users/jt/Desktop/left.json', '/Users/jt/Desktop/inner.json'

    CF = CommentFinder(None)
    df = CF.load_data_from_jsons(left)

    # do initial cleaning
    extras = ['subreddit_ann', 'subreddit_id', '_c0', 'body_html', 'id', 'id_ann', 'created',  'author_flair_text', 'author_flair_css_class', 'mod_reports']
    new_df = clean(df, extras)

    # split into train and test sets
    train, test1 = train_test_split(new_df, test_size=0.1)
    train, test2 = train_test_split(train, test_size=.25)


    # save smaller test set as test2, larger as test1
    test1.to_csv('test2.csv', header=True, encoding='utf-8')
    test2.to_csv('test1.csv', header=True, encoding='utf-8')
    train.to_csv('train.csv', header=True, encoding='utf-8')
