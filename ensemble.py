import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import recall_score, precision_score
from preprocessing import PreProcessor
import base_line_modeling as blm


class Ensemble(object):
    """
    Designed to help create a set of models, each successive model working on
    a subset of predictions of a previous model
    """
    def __init__(self, df_train, df_test, models):
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
        self.cached_df_check = False
        self.cached_label = None
        self.baseline_models = []
        self.baseline_scores = {}
        self.models = {}
        for i, model in enumerate(models):
             self.models[i+1]=model
        self.cv_scores = {}
        self.test_scores = None

    def create_model(self, targets, label_col='majority_type', score_type='precision'):
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
        print(self.df_train.columns)
        iteration_num = len(self.baseline_scores)+1
        if self.cached_df_check:
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
        print("Fitting model {} with {} features...".format(iteration_num, X_train.shape[1]))
        current.fit(X_train, y_train)

        # run baselines
        print("Fitting baseline models...")
        b, baselines = blm.run_baseline_modeling(bX_train, by_train, bX_test, by_test)

        self.baseline_models.append(b)
        self.baseline_scores[iteration_num] = baselines
        # # evaluate
        print("Evaluating models...")
        cv_score = cross_val_score(current, X_train, y_train, cv=3, scoring=score_type)
        self.cv_scores[iteration_num] = cv_score
        print("Baseline Scores: {}".format(baselines))
        print("Model {} CV Scores: ".format(iteration_num))
        for i, m in enumerate(self.cv_scores):
            print("{}: {}".format(m, self.cv_scores[m]))

        #update data
        df_train_2 = self._update(current, df_train_1, X_train, y_train)
        #for training, remove extraneous labels that switch through
        print("Caching updated and reduced data set for future passes")
        extras = set([k for k in targets if targets[k]])
        df_train_2 = self.remove_labels(df_train_2, extras)
        self.cached_df = df_train_2
        self.cached_df_check = True

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

    def run_test(self, df, targets_list, label_col='majority_type', score_type=['accuracy']):
        new = df.copy(deep=True)
        X, y = self._split(new, label_col)
        scores = {}
        baselines = {}
        for i, step in enumerate(targets_list):
            current = self.models[i+1]
            y_pred = current.predict(X)
            y = y.apply(lambda x: step[x])
            base = self.run_baselines(X, y, pass_n=i, targets=step)
            sc = []
            if 'accuracy' in score_type:
                sc.append(current.score(X, y))
            if 'precision' in score_type:
                sc.append(precision_score(y, y_pred))
            if 'recall' in score_type:
                sc.append(recall_score(y, y_pred))
            if i+2<=len(targets_list):
                new = new[new[label_col].isin(targets_list[i+1].keys())]
                X, y = self._split(new, label_col)
            print("Step {} Baseline Scores: {}".format(i+1, base))
            print("Step {} {} Scores : {}".format(i+1,score_type, sc))

            scores[i+1] = sc
            baselines[i+1]= base
        self.test_scores = scores
        return scores, baselines

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

    def run_baselines(self, X, y, pass_n='0', targets=None, label_col='majority_type'):
        """Runs baseline models in the train and test sets, storing their accuracies
        as [WeightedGuess_accuracy, 'GuessMostFrequent_accuracy'].
        X - features matrix
        y - labels matrix
        pass_n (string) - number appended to the storage name, allowing baselines to be run at different levels
        label_col (string) - column name containing target labels (y-values).
        """
        models = self.baseline_models[int(pass_n)]
        b_acc = []
        for m in models:
            pred = m.predict(X)
            pred = pd.Series(pred).apply(lambda x: step[x] if type(x) is not bool else x).values
            score = m.score(pred, y, scoring=['accuracy', 'recall', 'precision'])
            b_acc.append(score)
        print("Baseline scores on pass {}: {}".format(pass_n+1, b_acc))
        return b_acc

    # --------------- private helper functions below this line -------------- #
    def _preliminaries(self, targets, label_col='majority_type'):
        """Helper function for preliminary research on a particular step"""
        if self.cached_df_check:
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
        X = df.drop([label_col], axis=1)
        if old_label:
            self.cached_label = X[old_label]
            X = X.drop(old_label, axis=1)
        return X,y

    def _update(self, est, df, X, y, keep=[False]):
        """select portion of dataframe that contains keep"""
        # make predictions
        new_df = df.copy(deep=True)
        new_df['predicted'] = est.predict(X)
        # set aside the 'keeps'
        new_df = new_df[new_df['predicted'].isin(keep)]
        new_df = new_df.drop(['predicted', self.dummy_col], axis=1)
        return new_df


def data_prep(size=5000, train_file_path='data/train.csv', split=True, remove=None):
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

    if remove:
        remove_most = remove
    else:
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


def save_ensemble_results(ensemble, filename='results.txt', score_order=['accuracy', 'recall', 'precision']):
    with open(filename, 'w+') as f:
        f.write('Ensemble Results:\n\n')
        if hasattr(ensemble, 'models'):
            f.write('Models Used: \n')
            for model_num in ensemble.models:
                f.write("{}: {}\n".format(model_num, ensemble.models[model_num]))
        f.write('Models: {}\n'.format(ensemble.models))
        for result in ['baselines', 'scores', 'test_scores', 'base_line_tests']:
            if hasattr(ensemble, result):
                f.write("{}, in order: {}\n".format(result.capitalize(), score_order))
                for step in getattr(ensemble, result):
                    f.write("{} : {}\n ".format(step, getattr(ensemble, result)[step]))


if __name__ == '__main__':
    # load in data
    # Note: This is done from prepped data files. If these do not exist, use the following:

    # # in case prepped data doesn't exist:
    # df_train, df_test = data_prep(size=10000, train_file_path='data/train.csv', split=True, remove=None)
    # test_df = data_prep(size=2000, train_file_path='data/test1.csv', split=False)
    # print ("Data loaded.")


    df = pd.read_csv('prepped_train_data.csv')
    print ("Train data loaded.")
    df_train, df_test = train_test_split(df)
    test_df = pd.read_csv('prepped_test_data.csv')
    print ("Test data loaded.")


    # binary basic model testing setup (for to contrast with basic baselines, ensemble)
    straight_to_it = {'question': False, 'answer': False, 'elaboration': False, 'appreciation': False, 'agreement': False, 'disagreement': False, 'humor': False, 'negativereaction': True}
    basic_steps=[straight_to_it]
    try_models = [AdaBoostClassifier(n_estimators=400, learning_rate=0.2) ]



    # Use basic binary models as baselines. Print and save results
    for i, m in enumerate(try_models):
        FirstStep = Ensemble(df_train, df_test, [m])
        FirstStep.create_model(straight_to_it)
        print("Model created.\n")
   	    print("Running test on {} rows of Test data...".format(len(test_df)))
    	model_results, baseline_results = FirstStep.run_test(test_df, basic_steps, score_type=['accuracy', 'recall', 'precision'])

       	print("Results for model #{}".format(i))
       	for step in model_results:
            	print("{}:".format(step))
            	print("Model: {}".format(model_results[step]))
            	print("Baseline: {}".format(baseline_results[step]))
        save_ensemble(FirstStep, filename='basic_binary_model{}_results.txt'.format(i), score_order=['accuracy', 'recall', 'precision'])

    # Create and run my ensemble model
    direct_answers  = {'question': False, 'answer': True, 'elaboration': True, 'appreciation': False, 'agreement': False, 'disagreement': False, 'humor': False, 'negativereaction': False}
    discussion  = {'question': True, 'appreciation': True, 'agreement': True, 'disagreement': True, 'humor': False, 'negativereaction': False}
    tempers = {'humor': False, 'negativereaction': True}
    steps = [direct_answers, discussion, tempers]
    step_models = [GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3),
                AdaBoostClassifier(n_estimators=400, learning_rate=0.2),
                GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3)]

    myEnsemble = Ensemble(df_train, df_test, step_models)
    print("Training Ensemble on {} rows of data".format(len(df)))
    for s in steps:
        myEnsemble.create_model(s)
    print("Ensemble created.\n")

    test_df = data_prep(size=60000, train_file_path='data/test1.csv', split=False)
    print ("Test data loaded.")
    print("Running test on {} rows of Test data...".format(len(test_df)))
    model_results, baseline_results = myEnsemble.run_test(test_df, steps, score_type=['accuracy', 'recall', 'precision'])

    print("Results")
    for step in model_results:
        print("{}:".format(step))
        print("Model: {}".format(model_results[step]))
        print("Baseline: {}".format(baseline_results[step]))

    save_ensemble_results(myEnsemble, filename='ensemble_results.txt', score_order=['accuracy', 'recall', 'precision'])


    #for local Running
    # df_train_orig, df_test = data_prep(size=5000)
    # # test_df = data_prep(size=5000, train_file_path='data/test1.csv', split=False)
    # # print ("Data loaded.")
    #
    # df_train = df_train_orig.copy(deep=True)
    # FirstStep = Ensemble(df_train, df_test)
    #
    # df_1, X_train, df_2 = FirstStep.create_model(direct_answers)
    #
    # X, y = FirstStep._split(test_df, label_col='majority_type')
    # y_pred = FirstStep.models[1].predict(X)

    # FirstStep = Ensemble(df_train, df_test)
    # FirstStep.create_model(direct_answers)
    # FirstStep.create_model(discussion)
    # FirstStep.create_model(tempers)

    # ys = FirstStep.run_test(test_df, steps, score_type=['accuracy', 'recall', 'precision'])
    #save_ensemble(FirstStep, 'results_boost400_fulldata.txt')
