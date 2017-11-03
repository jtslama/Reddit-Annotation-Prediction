import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.metrics import recall_score, precision_score
from preprocessing import PreProcessor


class Baseline(object):
    """
    Blank Model for baseline models. Functions left intentionally blank.
    Initializes with train and test data.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, X, y):
        return None

    def predict(self, X):
        return None

    def score(self, y_pred, y, scoring=['accuracy']):
        """
        Calculates the accuracy of the model (#correctly classified/
        #incorrectly classified)
        INPUTS:
        y_pred (numpy array) - array of predicted labels of shape [n_samples, 1]
        y (numpy array) - array of actual labels of shape [n_samples, 1]
        scoring (string) - currently nonfunctional. Future implentation planned
                        for different accuracy metrics.
        OUTPUTS:
        acc (float) - accuracy of the model, as defined by the scoring parameter
        """
        acc = []

        if 'accuracy' in scoring:
            acc1 = np.sum(y_pred==y)*1.0/len(y)
            acc.append(acc1)
        if 'recall' in scoring:
            acc2 = recall_score(y, y_pred)
            acc.append(acc2)
        if 'precision' in scoring:
            acc3 = precision_score(y, y_pred)
            acc.append(acc3)
        return acc

    def run(self, score_type=['accuracy', 'recall', 'precision']):
        """Runs a basic accuracy test on the model"""
        self.fit(self.X_train, self.y_train)
        y_pred = self.predict(self.X_test)
        y_pred, self.y_test = y_pred.astype(np.bool_), self.y_test.astype(np.bool_)
        # print("y_pred shape, type: {}, {}".format(y_pred.shape, type(y_pred)))
        # print("y_pred uniques: {}".format(np.unique(y_pred)))
        # print("y_true shape, type: {}, {}".format(self.y_test.shape, type(y_pred)))
        # print("y_true uniques: {}".format(np.unique(self.y_test)))
        acc = self.score(y_pred, self.y_test, scoring=score_type)
        for i, s_type in enumerate(score_type):
            print("{} score of {}".format(s_type, acc[i]))
        return acc

    def run_test(self, X, y):
        y_pred = self.predict(self.X_test)


class WeightedGuess(Baseline):
    """
    Looks at the distribution of labels in the train data, and makes a random
    guess in proportion with the label distribution (e.g. if 40% of the labels
    in the train data are 'A', the model will guess A 40% of the time on average)
    """
    def fit(self, X, y):
        """
        Stores the unique label values as a list. Stores the relative proportion
        of each label as corresponding list.
        INPUTS:
        X - feature matrix. Traditional input to fit function, but unused here.
        y (1-dimensional pandas dataframe) - series containing labels
        """
        proportions = 1.0*y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.thresholds = proportions.values

    def predict(self,X):
        """
        Makes a random guess in proportion to the label distribution
        INPUTS:
        X (pandas dataframe or numpy array) - feature matrix of shape
                                            [n_samples, n_features]
        OUTPUT:
        y_pred (numpy array) - predicted label matrix of shape [n_features, 1]
        """
        y_pred = np.random.choice(self.labels, size=(X.shape[0],), p=self.thresholds)
        return y_pred


class MajorityGuess(WeightedGuess):
    """
    Looks at the distribution of labels in the train data, and guess the most
    frequently occurring label (e.g. if 60% of the labels in the train data are
    'A', the model will guess A every time).
    """
    def fit(self, X, y):
        """
        Stores the unique label values as a list. Finds the label with the
        highest relative frequency
        INPUTS:
        X - feature matrix. Traditional input to fit function, but unused here.
        y (1-dimensional pandas dataframe) - series containing labels
        """
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.guess = np.argmax(proportions)

    def predict(self, X):
        """
        Makes a random guess in proportion to the label distribution
        INPUTS:
        X (pandas dataframe or numpy array) - feature matrix of shape
                                            [n_samples, n_features]
        OUTPUT:
        y_pred (numpy array) - predicted label matrix of shape [n_features, 1]
        """
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.guess)
        return y_pred

def run_baseline_modeling(X_train, y_train, X_test, y_test):
    """Function to instantiate, fit, and score the baseline models for the given
    data. Used to benchmark other models.
    INPUTS:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    X_test (pandas dataframe) - test features matrix of shape
                                [x_samples, n_features]
    y_test (pandas dataframe) - test labels matrix of shape [x_samples, 1]
    OUTPUTS:
    baselines (list of floats) - list containing accuracy scores for each
                                baseline model in order of: WeightedGuess,
                                MajorityGuess
    """
    # establish baseline models
    print("Running baseline models...")
    WG = WeightedGuess(X_train, y_train, X_test, y_test)
    print("Weighted Guess scores: ")
    WG_acc = WG.run(score_type=['accuracy', 'recall', 'precision'])
    MJ = MajorityGuess(X_train, y_train, X_test, y_test)
    print("Majority Guess scores: ")
    MJ_acc = MJ.run(score_type=['accuracy', 'recall', 'precision'])
    models = [WG, MJ]
    baselines = [WG_acc, MJ_acc]
    return models, baselines


def run_basic_nb_models(X_train, y_train, X_test, y_test, score_type='accuracy'):
    """
    Helper function to run Naive Bayes Model on the data. Multinomial Naive
    Bayes not yet implemented (would require different matrix factorizations
    than those currently in use).
    INPUTS:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    X_test (pandas dataframe) - test features matrix of shape
                                [x_samples, n_features]
    y_test (pandas dataframe) - test labels matrix of shape [x_samples, 1]
    OUTPUTS:
    GNB (object) - fitted Naive Bayes model
    GNB_scores (float) - GNB model accuracy score
    """
    print("Running Naive Bayes models...")
    GNB = GaussianNB().fit(X_train, y_train)
    GNB_scores = cross_val_score(GNB, X_test, y=y_test, cv=5, n_jobs=-1, scoring=score_type)
    print("Bayes scores:\n{}".format(GNB_scores))
    return GNB, GNB_scores

def run_alt_model_tests(X_train, y_train, score_type='accuracy'):
    """
    Helper function for running grid searches on varying models and parameters,
    prints best results. Mostly for exploration.
    INPUT:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    OUTPUT:
    None
    """
    #look at some basic model results:
    print("Beginning grid search...")
    print("Running grid search on LogReg...")
    lr = LogisticRegressionCV()
    LRparams = {'solver': ['newton-cg']}
    lr_gs = GridSearchCV(lr, LRparams, n_jobs=-1, scoring=score_type).fit(X_train, y_train)
    lr_res = [lr_gs.best_score_, lr_gs.best_params_]
    print("LogReg desired scores and params\n{}\n{}".format(lr_res[0], lr_res[1]))

    print("Running grid search on RF...")
    rf = RandomForestClassifier()
    RFparams = {'criterion': ['entropy'],
                'n_estimators': [5000],
                'max_features': [None]}
    rf_gs = GridSearchCV(rf, RFparams, n_jobs=-1,scoring=score_type).fit(X_train, y_train)
    rf_res = [rf_gs.best_score_, rf_gs.best_params_]
    print("RF desired scores and params\n{}\n{}".format(rf_res[0], rf_res[1]))
    print("Running grid search on GBC...")
    est = GradientBoostingClassifier()
    GBparams = {'n_estimators': [400,600,800],
                'learning_rate': [.1, .05, .02, .01]}
    gb_gs = GridSearchCV(est, GBparams, n_jobs=-1, scoring=score_type).fit(X_train, y_train)
    gs_res = [gb_gs.best_score_, gb_gs.best_params_]
    print("GBC desired scores and params\n{}\n{}".format(gs_res[0], gs_res[1]))
    print("Running grid search on Adaboost...")
    ada = AdaBoostClassifier()
    ABparams = {'n_estimators': [400,600,800],
                'learning_rate': [.1, .05, .02, .01]}
    ab_gs = GridSearchCV(ada, ABparams, n_jobs=-1, scoring=score_type).fit(X_train, y_train)
    ab_res = [ab_gs.best_score_, ab_gs.best_params_]
    print("ABC desired scores and params\n{}\n{}".format(ab_res[0], ab_res[1]))

    print("\nFinal Results:")
    # print("LogReg desired scores and params\n{}\n{}".format(lr_res[0], lr_res[1]))
    # print("RF desired scores and params\n{}\n{}".format(rf_res[0], rf_res[1]))
    print("GBC desired scores and params\n{}\n{}".format(gs_res[0], gs_res[1]))
    print("ABC desired scores and params\n{}\n{}".format(ab_res[0], ab_res[1]))
    # return [lr_gs, rf_gs, gb_gs, ab_gs]


def run_alt_models(X_train, y_train, X_test, y_test):
    """
    Designed to compare specific models to one another.
    INPUTS:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    X_test (pandas dataframe) - test features matrix of shape
                                [x_samples, n_features]
    y_test (pandas dataframe) - test labels matrix of shape [x_samples, 1]
    OUTPUTS:
    models (list of objects) - list of fitted models
    scores (list of float) - list of model accuracy scores (in the same order)
    """
    #look at some basic model results:
    print("Running LogReg...")
    LR = LogisticRegressionCV(n_jobs=-1, cv=10, solver='newton-cg').fit(X_train, y_train)
    LR_acc = LR.score(X_test, y_test)
    print("LogReg accuracy score: {}".format(LR_acc))


    print("Running RandomForest...")
    RF = RandomForestClassifier(n_estimators=10000, n_jobs=-1).fit(X_train, y_train)
    RF_acc = RF.score(X_test, y_test)
    print("Random Forest accuracy score: {}".format(RF_acc))

    print("Running GradientBoost...")
    GBC = GradientBoostingClassifier(n_estimators=500, learning_rate=.02, max_depth=3).fit(X_train, y_train)
    GBC_acc = GBC.score(X_test, y_test)
    GBparams = {'n_estimators': [10,100,500],
                'learning_rate': [.1, .05, .02],
                'max_depth': [2, 3, 5]}
    print("GBC accuracy score: {}".format(GBC_acc))

    print("Running Adaboost...")
    ADA = AdaBoostClassifier(n_estimators=500, learning_rate=0.1).fit(X_train, y_train)
    ADA_acc = ADA.score(X_test, y_test)
    print("Adaboost accuracy score: {}".format(ADA_acc))

    scores = [LR_acc, RF_acc, GBC_acc, ADA_acc]
    models = [LR, RF, GBC, ADA]
    return models, scores


def main(size=5000, grid=False):
    """
    Composite function designed for running tests.
    INPUTS:
    size (int) - number of rows of the data set to use
    grid (bool) - whether or not to grid search
    OUTPUTS:
    None
    """
    # prepare data for modeling
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    if size > len(train):
        df = train
    df= train[:size]

    #make splits
    print("Splitting...")
    df_train, df_test = train_test_split(df, test_size=0.20)

    print("Preprocessing...")
    P = PreProcessor()
    remove_all_but_text = None

    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited','gilded', 'in_reply_to',
                   'is_first_post', 'link_id', 'link_id_ann', 'majority_link', \
                   'name', 'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    X_train, y_train = P.run(df_train, 'body', cols_to_drop=remove_most, direct_to_model=True)
    X_test, y_test = P.run(df_test, 'body', cols_to_drop=remove_most, direct_to_model=True)

    # establish baseline models
    baseline_scores = run_baseline_modeling(X_train, y_train, X_test, y_test)
    # look at basic NB model results (reduced to NB)
    nb_models, NB_base_scores = run_basic_nb_models(X_train, y_train, X_test, y_test)

    if grid:
        #run grid search
        run_alt_model_tests(X_train, y_train, X_test, y_test)
    else:
        # look at basic model scores
        alt_models, alt_scores = run_alt_models(X_train, y_train, X_test, y_test)
        print("\n\nBaseline Scores: ")
        for n, s in zip(['Weighted Guess', 'Guess Most Frequent'], baseline_scores):
            print("{}: {}".format(n,s))
        print("Naive Bayes Scores")
        for n, s in zip(['Naive Bayes', 'Multinomial Bayes'], NB_base_scores):
            print("{}: {}".format(n,s))
        print("Other model Scores: ")
        for n, s in zip(['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Adaboost'], alt_scores):
            print("{}: {}".format(n,s))



if __name__ == '__main__':
    main(size=5000, grid=True)
