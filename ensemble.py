import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import PreProcessor, graph_label_distribs
import base_line_modeling as blm
# import matplotlib.pyplot as plt



class Ensemble(object):
    """For customizing the labels I use in modeling"""
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.grids = []
        self.models = []

    def remove_labels(self, df, labels, label_col):
        """
        INPUT:
        df (pandas dataframe) - dataframe being filtered
        labels (list) - labels to be removed
        col - column name labels are being selected in
        OUTPUT:
        out (pandas dataframe) - dataframe with rows that have labels in col
                                removed
        """
        out = df[~df[label_col].isin(labels)]
        return out

    def change_labels(self, df, targets, label='majority_type'):
        """targets is a dict pairing old labels to their new labels
        targets = {'old_label1': 'new_label1', 'old_label2': 'new_label2'}
        """
        new_target = df[label].apply(lambda x: targets[x]if x in targets else x)
        output = df
        output[label] = new_target
        return output

    def run_grid(self):
        X_train, y_train = self._split(self.df_train, 'majority_type')
        X_test, y_test = self._split(self.df_test, 'majority_type')
        baseline_scores = blm.run_baseline_modeling(X_train, y_train, X_test, y_test)
        self.grids = blm.run_alt_model_tests(X_train, y_train, X_test, y_test)
        print("\nBaseline model scores for comparison:\n{}".format(baseline_scores))

    def _split(self, df, label_col):
        y = df[label_col]
        X = df.drop(label_col, axis=1)
        return X,y

    def preliminaries(self, targets, label_col='majority_type'):
        self.df_train = self.change_labels(self.df_train, targets, label=label_col)
        self.df_test = self.change_labels(self.df_test, targets, label=label_col)
        self.run_grid()


def _prep(df, size=5000, split=False):
    subset = df[:size]
    print("Testing on {} rows...".format(len(subset)))
    P = PreProcessor()
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited','gilded', 'in_reply_to',
                   'is_first_post', 'link_id', 'link_id_ann', 'majority_link', \
                   'name', 'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']
    if split:
        # make splits
        print("Splitting...")
        df_train, df_test = train_test_split(df, test_size=0.20)
        print("Preprocessing...")
        X_train, y_train = P.run(df_train, 'body', cols_to_drop=remove_most, direct_to_model=True)
        X_test, y_test = P.run(df_test, 'body', cols_to_drop=remove_most, direct_to_model=True)
        return df_train, df_test
    else:
    # don't split
        df_train = P.run(subset, 'body', cols_to_drop=remove_most)
        print('df_train', type(df_train))
        return df_train


def _test(df, size=5000):
    P = Ensemble()
    #remove 'other'
    df = P.remove_labels(df, ['other'], 'majority_type')
    #try changing label paradigm
    support_disrupt = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'disrupt', 'humor': 'disrupt', 'negativereaction': 'disrupt'}
    response_digression = {'question': 'response', 'answer': 'response', 'elaboration': 'response', 'appreciation': 'digression', 'agreement': 'response', 'disagreement': 'digression', 'humor': 'digression', 'negativereaction': 'digression'}
    conflict = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'disrupt', 'humor': 'support', 'negativereaction': 'disrupt'}
    negative_rxn = {'question': 'pos', 'answer': 'pos', 'elaboration': 'pos', 'appreciation': 'pos', 'agreement': 'pos', 'disagreement': 'neutral', 'humor': 'neutral', 'negativereaction': 'negative'}
    df_support = P.change_labels(df, negative_rxn)

    return df_support


def data_prep(size=5000, train_file_path='data/train.csv'):
    # prepare data for modeling
    print("Loading data...")
    train = pd.read_csv(train_file_path)
    if size > len(train):
        df = train
    df= train[:size]

    #make splits
    print("Splitting...")
    df_train, df_test = train_test_split(df, test_size=0.25)

    print("Preprocessing...")
    P = PreProcessor()
    remove_all_but_text = None

    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited','gilded', 'in_reply_to',
                   'is_first_post', 'link_id', 'link_id_ann', 'majority_link', \
                   'name', 'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    df_train = P.run(df_train, 'body', cols_to_drop=remove_most, direct_to_model=False)
    df_test = P.run(df_test, 'body', cols_to_drop=remove_most, direct_to_model=False)
    return df_train, df_test


if __name__ == '__main__':
    #ideas for an ensemble
    relevant_ans = {'question': 'notanswer', 'answer': 'direct', 'elaboration': 'direct', 'appreciation': 'notanswer', 'agreement': 'notanswer', 'disagreement': 'notanswer', 'humor': 'notanswer', 'negativereaction': 'notanswer'}
    argument = {'question': 'neutral', 'answer': 'other', 'elaboration': 'other', 'appreciation': 'positive', 'agreement': 'positive', 'disagreement': 'negative', 'humor': 'neutral', 'negativereaction': 'negative'}

    df_train, df_test = data_prep(size=60000)
    FirstStep = Ensemble(df_train, df_test)
    FirstStep.preliminaries(relevant_ans)









    #other ideas
    support_disrupt = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'disrupt', 'humor': 'disrupt', 'negativereaction': 'disrupt'}
    response_digression = {'question': 'response', 'answer': 'response', 'elaboration': 'response', 'appreciation': 'digression', 'agreement': 'response', 'disagreement': 'digression', 'humor': 'digression', 'negativereaction': 'digression'}
    conflict = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'disrupt', 'humor': 'support', 'negativereaction': 'disrupt'}
    negative_rxn = {'question': 'pos', 'answer': 'pos', 'elaboration': 'pos', 'appreciation': 'pos', 'agreement': 'pos', 'disagreement': 'neutral', 'humor': 'neutral', 'negativereaction': 'negative'}
