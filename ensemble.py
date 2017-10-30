import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from preprocessing import PreProcessor
import base_line_modeling as blm



class Ensemble(object):
    """
    Designed to help create a set of models, each successive model working on
    a subset of predictions of a previous model
    """
    def __init__(self, df_train, df_test):
        """
        INPUTS:
        df_train (pandas dataframe) - preprocessed training data of shape
                                    [n_samples, m_features]
        df_test (pandas dataframe) - preprocessed test data of shape
                                    [x_samples, m_features]
        """
        self.df_train = df_train
        self.df_test = df_test
        self.dummy_col = 'is_p'
        self.cached_df = None
        self.cached_label = None
        self.baselines = {}
        self.models = {1: GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3),
                    2: AdaBoostClassifier(n_estimators=400, learning_rate=0.2)}
        self.scores = {}

    def create_model(self, targets, label_col='majority_type'):
        """
        Use target parameters to create a new label system. Train a model on
        the new target parameters, and use its predictions to create a new
        subset of data.
        INPUTS:
        targets (dict) - keys are current labels, values are their new
                        designation; e.g. 'question': False
        label_col (string) - name of the column holding the old labels used to
                            translate (column containing the keys in targets)
        OUTPUTS:
        None
        """
        iteration_num = len(self.baselines)+1
        if self.cached_df:
            df_train_1 = self.cached_df
        else:
            df_train_1 = self.df_train
        #update labels
        print("Setting up targets...")
        df_train_1 = self.change_labels(df_train_1, targets, label=label_col, new_col=self.dummy_col)
        X_train, y_train = self._split(df_train_1, self.dummy_col, old_label='majority_type')
        bX_train, bX_test, by_train, by_test = train_test_split(X_train, y_train)
        #choose model
        current = self.models[iteration_num]
        # LR = LogisticRegression(solver='newton-cg')
        # RF = RandomForestClassifier(n_estimators=10000, max_features=None, criterion='entropy')
        print("Training and evaluating model...")
        current.fit(X_train, y_train)
        # run baselines
        baselines = blm.run_baseline_modeling(bX_train, by_train, bX_test, by_test)
        self.baselines[iteration_num] = baselines
        #evaluate
        cv_score = cross_val_score(current, X_train, y_train, cv=10)
        self.scores[iteration_num] = cv_score
        print("Baseline Scores: {}".format(baselines))
        print("CV Scores: ")
        for i, m in enumerate(self.models):
            print("{}: {}".format(m, cvs[i]))
        #update data
        df_train_2 = self._update(current, df_train_1, X_train, y_train)
        #for training, remove extraneous labels that switch through
        print("Caching updated and reduced data set for future passes")
        old_labels = set(df_train[label_col].unique())
        extras = set(targets.keys())
        extras.difference_update(current_labels)
        df_train_2 = self.remove_labels(df_train_2, extras)
        self.cached_df = df_train_2

    def remove_labels(self, df, discard, label_col='majority_type'):
        """
        Remove data points if their names are in discard. Used for training the
        models.
        INPUT:
        df (pandas dataframe) - dataframe being filtered
        discard (list of strings) - list of labels to be removed
        label_col - column name labels are being selected in
        OUTPUT:
        out (pandas dataframe) - dataframe with rows that have labels in col
                                removed
        """
        out = df[~df[label_col].isin(discard)]
        return out

    def change_labels(self, df, targets, label='majority_type', new_col='is_p'):
        """
        Adds a column to a dataframe according to target parameters
        INPUTS:
        df (pandas dataframe) - dataframe containing labels column to be
                                translated
        targets (dict) - pairs original labels (keys) to their new labels (values)
                        e.g. {'old_label1': 'new_label1' }
        label (string) - name of the column containing the original labels
        new_col (string) - name of new column containing the translated labels
        OUTPUTS:
        output (pandas dataframe) - dataframe containing original and newly
                                    translated columns
        """
        new_target = df[label].apply(lambda x: targets[x]if x in targets else x)
        output = df
        output[new_col] = new_target
        return output

    def run_baselines(self, pass_n='1', label_col='majority_type'):
        """Runs baseline models in the train and test sets, storing their accuracies
        as [WeightedGuess_accuracy, 'GuessMostFrequent_accuracy'].
        pass_n (string) - number appended to the storage name, allowing baselines to be run at different levels
        label_col (string) - column name containing target labels (y-values).
        """
        X_train, y_train = self._split(self.df_train, label_col)
        X_test, y_test = self._split(self.df_test, label_col)
        baseline_accuracies = blm.run_baseline_modeling(X_train, y_train, X_test, y_test)
        self.scores['baselines_{}'.format(pass_n)] = baseline_accuracies
        print("Baseline scores on pass {}: {}".format(pass_n, baseline_accuracies))
    # --------------- private helper functions below this line -------------- #
    def _preliminaries(self, targets, label_col='majority_type'):
        """Helper function for preliminary research on a particular step"""
        if self.cached_df:
            df = self.cached_df
        else:
            df = self.change_labels(self.df_train, targets, label=label_col)
            df_test = self.change_labels(self.df_test, targets, label=label_col)
        self.df_train.drop([label_col], axis=1, inplace=True)
        self._run_grid(df, label_col=self.dummy_col)

    def _run_grid(self, df, label_col='majority_type'):
        """
        For grid searching on a particular stage of the ensemble.
        """
        X_train, y_train = self._split(df, label_col)
        blm.run_alt_model_tests(X_train, y_train)

    def _split(self, df, label_col, old_label=None):
        y = df[label_col]
        X = df.drop(label_col, axis=1)
        if old_label:
            self.cached_label = X[old_label]
            X = X.drop(old_label, axis=1)
        return X,y

    def _update(self, est, df, X, y, keep=[False]):
        """select portion of dataframe that contains keep"""
        # add column for the log_probability of the predicted values
        X['log_prob'] = np.nanmax(est.predict_log_probaba(X), axis=0)
        # make predictions
        df['predicted'] = est.predict(X)
        # set aside the 'keeps'
        new_df = df[df['predicted'].isin(keep)]
        new_df = new_df.drop(['predicted', self.dummy_col], axis=1)
        return new_df


def data_prep(size=5000, train_file_path='data/train.csv', split=True):
    """
    Data preprocessing helper function for local running of the ensemble.
    INPUTS:
    size (int) - number of rows of the train data to use
    train_file_path (string) - filepath to location of train data (as csv)
    split (bool) - whether to split the data into train and test components or
                    leave as one unit.
    """
    # prepare data for modeling
    print("Loading data...")
    train = pd.read_csv(train_file_path)
    if size > len(train):
        df = train
    df= train[:size]

    print("Preprocessing...")
    P = PreProcessor()
    remove_all_but_text = None

    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited','gilded', 'in_reply_to',
                   'is_first_post', 'link_id', 'link_id_ann', 'majority_link', \
                   'name', 'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    if split:
        # make splits
        print("Splitting...")
        df_train, df_test = train_test_split(df, test_size=0.25)
        df_train = P.run(df_train, 'body', cols_to_drop=remove_most, direct_to_model=False)
        df_test = P.run(df_test, 'body', cols_to_drop=remove_most, direct_to_model=False)
        return df_train, df_test
    else:
        df_train = P.run(df, 'body', cols_to_drop=remove_most, direct_to_model=False)
        return df_train


if __name__ == '__main__':
    #ideas for an ensemble
    relevant_ans = {'question': 'notanswer', 'answer': 'direct', 'elaboration': 'direct', 'appreciation': 'notanswer', 'agreement': 'notanswer', 'disagreement': 'notanswer', 'humor': 'notanswer', 'negativereaction': 'notanswer'}
    argument = {'question': 'neutral', 'answer': 'other', 'elaboration': 'other', 'appreciation': 'positive', 'agreement': 'positive', 'disagreement': 'negative', 'humor': 'neutral', 'negativereaction': 'negative'}

    #other set
    direct_answers  = {'question': False, 'answer': True, 'elaboration': True, 'appreciation': False, 'agreement': False, 'disagreement': False, 'humor': False, 'negativereaction': False}
    discussion  = {'question': True, 'appreciation': True, 'agreement': True, 'disagreement': True, 'humor': False, 'negativereaction': False}
    tempers = {'humor': False, 'negativereaction': True}

    #alts
    on_topic = {'question': 'y', 'answer': 'y', 'elaboration': 'y', 'appreciation': 'y', 'agreement': 'c', 'disagreement': 'c', 'humor': 'c', 'negativereaction': 'c'}
    neg_2 = {'agreement': 'p', 'disagreement': 'p', 'humor': 'p', 'negativereaction': 'n'}
    #try changing label paradigm
    support_disrupt = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'support', 'humor': 'disrupt', 'negativereaction': 'disrupt'}
    response_digression = {'question': 'response', 'answer': 'response', 'elaboration': 'response', 'appreciation': 'digression', 'agreement': 'response', 'disagreement': 'digression', 'humor': 'digression', 'negativereaction': 'digression'}
    conflict = {'question': 'support', 'answer': 'support', 'elaboration': 'support', 'appreciation': 'support', 'agreement': 'support', 'disagreement': 'disrupt', 'humor': 'support', 'negativereaction': 'disrupt'}
    negative_rxn = {'question': 'pos', 'answer': 'pos', 'elaboration': 'pos', 'appreciation': 'pos', 'agreement': 'pos', 'disagreement': 'neutral', 'humor': 'neutral', 'negativereaction': 'negative'}



    # prepped_data = data_prep(size=60000, train_file_path='data/train.csv', split=False)
    # prepped_data.to_csv('prepped_train_data.csv')

    df = pd.read_csv('prepped_train_data.csv')
    # df = df[:5000]
    print ("Data loaded.")
    df_train, df_test = train_test_split(df)
    FirstStep = Ensemble(df_train, df_test)
    FirstStep.create_model(direct_answers)
    FirstStep._preliminaries(discussion)


    # #for local Running
    # df_train, df_test = data_prep(size=5000)
    # print ("Data loaded.")
    # FirstStep = Ensemble(df_train, df_test)
    # FirstStep.create_model(direct_answers)
    # FirstStep.create_model(discussion)
    # FirstStep.create_model(tempers)
