import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split,  cross_val_score
from preprocessing import PreProcessor


class Baseline(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, X, y):
        return None

    def predict(self, X):
        return None

    def score(self, y_pred, y, scoring='accuracy'):
        return None

    def run(self):
        self.fit(self.X_train, self.y_train)
        y_pred = self.predict(self.X_test)
        acc = self.score(y_pred, self.y_test)
        print("Accuracy score of {}".format(acc))
        return acc


class WeightedGuess(Baseline):

    def fit(self, X, y):
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.thresholds = proportions.values

    def predict(self,X):
        y_pred = np.random.choice(self.labels, size=(X.shape[0],), p=self.thresholds)
        return y_pred

    def score(self, y_pred, y_true, scoring='accuracy'):
        result = np.sum(y_pred==y_true)*1.0/len(y_true)
        return result

class MajorityGuess(WeightedGuess):

    def fit(self, X, y):
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.guess = np.argmax(proportions)

    def predict(self, X):
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.guess)
        return y_pred

def run_baseline_modeling(X_train, y_train, X_test, y_test):
    #basic guesses
    # establish baseline models
    print("Running baseline models...")
    WG_acc = WeightedGuess(X_train, y_train, X_test, y_test).run()
    MJ_acc = MajorityGuess(X_train, y_train, X_test, y_test).run()
    baselines = [WG_acc, MJ_acc]
    print("Baseline scores: {}".format(baselines))
    return baselines


def run_basic_nb_models(X_train, y_train, X_test, y_test):
    #NB
    print("Running Naive Bayes models...")
    GNB = GaussianNB().fit(X_train, y_train)
    # MNB = MultinomialNB().fit(X_train, y_train)
    GNB_scores = cross_val_score(GNB, X_test, y=y_test, cv=5, n_jobs=-1)
    # MNB_scores = cross_val_score(MNB, X_test, y=y_test, cv=5, n_jobs=-1)
    # Bayes_scores = [GNB_scores, MNB_scores]
    # models = [GNB, MNB]
    # print("Bayes scores:\n{}".format(Bayes_scores))
    # return models, Bayes_scores
    print("Bayes scores:\n{}".format(GNB_scores))
    return GNB, GNB_scores

def run_alt_model_tests(X_train, y_train):
    #look at some basic model results:
    print("Beginning grid search...")
    # print("Running grid search on LogReg...")
    # lr = LogisticRegressionCV()
    # LRparams = {'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag']}
    # lr_gs = GridSearchCV(lr, LRparams, n_jobs=-1).fit(X_train, y_train)
    # lr_res = [lr_gs.best_score_, lr_gs.best_params_]
    # print("LogReg desired scores and params\n{}\n{}".format(lr_res[0], lr_res[1]))
    #
    # print("Running grid search on RF...")
    # rf = RandomForestClassifier()
    # RFparams = {'criterion': ['gini', 'entropy'],
    #             'max_features': ["auto", "sqrt", "log2", None]}
    # rf_gs = GridSearchCV(rf, RFparams, n_jobs=-1, ).fit(X_train, y_train)
    # rf_res = [rf_gs.best_score_, rf_gs.best_params_]
    # print("RF desired scores and params\n{}\n{}".format(rf_res[0], rf_res[1]))
    print("Running grid search on GBC...")
    est = GradientBoostingClassifier()
    GBparams = {'n_estimators': [50,100,200,400],
                'learning_rate': [.25, .2, .15, .1, .05]}
    gb_gs = GridSearchCV(est, GBparams, n_jobs=-1).fit(X_train, y_train)
    gs_res = [gb_gs.best_score_, gb_gs.best_params_]
    print("GBC desired scores and params\n{}\n{}".format(gs_res[0], gs_res[1]))
    print("Running grid search on Adaboost...")
    ada = AdaBoostClassifier()
    ABparams = {'n_estimators': [50,100,200,400],
                'learning_rate': [.25, .2, .15, .1, .05]}
    ab_gs = GridSearchCV(ada, ABparams, n_jobs=-1).fit(X_train, y_train)
    ab_res = [ab_gs.best_score_, ab_gs.best_params_]
    print("ABC desired scores and params\n{}\n{}".format(ab_res[0], ab_res[1]))

    print("\nFinal Results:")
    # print("LogReg desired scores and params\n{}\n{}".format(lr_res[0], lr_res[1]))
    # print("RF desired scores and params\n{}\n{}".format(rf_res[0], rf_res[1]))
    print("GBC desired scores and params\n{}\n{}".format(gs_res[0], gs_res[1]))
    print("ABC desired scores and params\n{}\n{}".format(ab_res[0], ab_res[1]))
    # return [lr_gs, rf_gs, gb_gs, ab_gs]


def run_alt_models(X_train, y_train, X_test, y_test):
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
        #run gridsearch
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
    main(size=10000, grid=True)





    """
    # prepare data for modeling
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    df = train[:2000]
    df.head()


    print("Splitting...")
    df_train, df_test = train_test_split(df, test_size=0.25)

    print("Preprocessing...")
    P = PreProcessor()
    remove_all_but_text = None

    #TODO: deal with in_reply_to, parent_id
    #is_first_post is a mix of false, nan and annotation values
    #subreddit --> convert somehow?
    #title > metric for how often words appear in reply
    #gilded: temp removed because it's causing bugs
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited','gilded', 'in_reply_to', 'is_first_post', \
                   'link_id', 'link_id_ann', 'majority_link', 'name',  \
                   'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', 'ann_1', 'ann_2', 'ann_3']

    X_train, y_train = P.run(df_train, 'body', cols_to_drop=remove_most, direct_to_model=True)
    X_test, y_test = P.run(df_test, 'body', cols_to_drop=remove_most, direct_to_model=True)


    # look at basic NB model results
    nb_models, NB_base_scores = run_basic_nb_models(X_train, y_train, X_test, y_test)
    #TODO throwing error: invalid data type (infin, Nan or too big)

    # establish baseline models
    baseline_models, baseline_scores = run_baseline_modeling(X_train, y_train)

    # alternative model ideas
    # alt_models = run_alt_model_ideas(X_train, y_train, X_test, y_test)
    alt_models, alt_scores = run_alt_model_tests(X_train, y_train)
    """
