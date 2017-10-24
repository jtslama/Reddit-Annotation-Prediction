import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split,  cross_val_score
from preprocessing import Processor


class WeightedGuess(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.thresholds = proportions.values

    def predict(self,X):
        y_pred = np.random.choice(self.labels, size=(X.shape[0],), p=self.thresholds)
        return y_pred

    def score(self, y_pred, y_true):
        accuracy = np.sum(y_pred==y_true)*1.0/len(y_true)
        return accuracy

class MajorityGuess(WeightedGuess):

    def fit(self, X, y):
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values
        self.guess = np.argmax(proportions)

    def predict(self, X):
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.guess)
        return y_pred

def baseline_modeling(X_train, y_train):
    #basic guesses
    # establish baseline models
    print("Running baseline models...")
    WG = WeightedGuess()
    WG.fit(X_train, y_train)
    WG_pred = WG.predict(X_train)
    # WG_pred.shape
    # y_train.shape
    WG_acc = WG.score(WG_pred, y_train)

    MJ = MajorityGuess()
    MJ.fit(X_train,y_train)
    MJ_pred = MJ.predict(X_train)
    MJ_acc = MJ.score(MJ_pred, y_train)
    models = [WG, MJ]
    baselines = [WG_acc, MJ_acc]
    print("Baseline scores: {}".format(baselines))
    return models, baselines


def basic_nb_models(X_train, y_train):
    #NB
    print("Running Naive Bayes models...")
    GNB = GaussianNB().fit(X_train, y_train)
    MNB = MultinomialNB().fit(X_train, y_train)
    GNB_scores = cross_val_score(GNB, X_train, y=y_train, cv=5, n_jobs=-1)
    MNB_scores = cross_val_score(MNB, X_train, y=y_train, cv=5, n_jobs=-1)
    Bayes_scores = [GNB_scores, MNB_scores]
    models = [GNB, MNB]
    print("Bayes scores:\n{}".format(Bayes_scores))
    return models, Bayes_scores


if __name__ == '__main__':
        # prepare data for modeling
        print("Loading data...")
        train = pd.read_csv('data/train.csv')

        print("Preprocessing...")
        processor = Processor()
        content = processor.prepare_data(train)

        print("Splitting...")
        X_train, X_test, y_train, y_test = train_test_split(content['body'], content['majority_type'], test_size=0.25)
        X_train, vocab = processor.vectorize(X_train)

        
        # establish baseline models
        models, baseline_scores = baseline_modeling(X_train, y_train)

        # look at basic NB model results
        models, NB_base_scores = basic_nb_models(X_train, y_train)


        #look at some basic model results:
        print("Running grid search on GBC...")
        est = GradientBoostingClassifier()
        GBparams = {'learning_rate': [.1, .05, .02],
                  'max_depth': [2, 3, 5],
                  'min_samples_leaf': [2, 3, 5, 10]}
        gb_gs = GridSearchCV(est, GBparams, n_jobs=-1).fit(X_train, y_train)
        gs_res = [gb_gs.best_score_, gb_gs.best_params_]
        print("GBC desired scores and params\n{}\n{}".format(gs_res[0], gs_res[1]))
        print("Running grid search on RF,,,")
        rf = RandomForestClassifier()
        RFparams = {'n_estimators': np.linspace(50,1000)}
        rf_gs = GridSearchCV(rf, RFparams, n_jobs=-1).fit(X_train, y_train)
        rf_res = [rf_gs.best_score_, rf_gs.best_params_]
        print("RFdesired scores and params\n{}\n{}".format(rf_res[0], rf_res[1]))

        #sklearn.metrics.r2score



        #y_train.unique()
