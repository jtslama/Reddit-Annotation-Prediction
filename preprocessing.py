import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import defaultdict



class PreProcessor(object):
    """Primarily for cleaning the data and converting the comment bodies to tfidf
    vectors
    """
    def __init__(self):
        self._make_consensus = False
        self.SVD = None
        self.tfidf = None

    def matrix_reduction(self, matrix, n_comp=2000, power_level=1):
        """
        INPUTS:
        matrix(numpy array) - the matrix to be reduced
        n_comp (int) - dimension to initially reduce to, must be less than
                        number of features (matrix.shape[1])
        power_level(float) - the percentage of explained variance desired
                            (b/t 0 and 1) if extra reduction is desired
        OUTPUTS:
        U_trunc - the trunkated user weight array (rows: items, columns:
                    latent features, values: weights)
        Sigma_trunc - the trunkated power array (rows: latent features,
                        columns: power, values: power)
        """
        initial_size = matrix.shape
        if power_level>1:
            print("No. Power level can't be more than 1. Setting power_level to 0.9")
            power_level=0.9
        if n_comp >= matrix.shape[1]:
            print("n_comp must be between 2 and the original number of features. Setting to half of original features")
            n_comp = min(2000, matrix.shape[1]-1)
        #decompose matrix into U,S,V components
        #fit the SVD model the first time around (use train data)
        if not self.SVD:
            print("Creating initial TruncatedSVD fit...")
            self.SVD = TruncatedSVD(n_components=n_comp, n_iter=10).fit(matrix)
        U_trunc = self.SVD.transform(matrix)
        Sigma_sq = self.SVD.explained_variance_ratio_
        p = Sigma_sq[:U_trunc.shape[1]].sum()
        middle_size = U_trunc.shape
        #shrink matrix to latent features that account for power_level fraction of total power
        print("Shrunk tfidf vectors from {} features to {}, preserving {}% of explained variance".format(initial_size[1], middle_size[1], 100*p))
        if power_level:
            i = self._find_desired_power(Sigma_sq, power_level)
            U_trunc = U_trunc[:, :i]
            final_size = U_trunc.shape
            print("Further shrunk tfidf vectors from {} features to {}, preserving {}% of explained variance".format(middle_size[1], final_size[1], 100*power_level))

        return U_trunc, Sigma_sq

    def prepare_data(self, df, remove=None):
        """
        A series of customized steps to clean the raw data. Includes removing
        unwanted columns, removing, fixing or deleting data with nulls
        (dependent on category).
        INPUTS:
        df (pandas dataframe) - the raw dataframe to be cleaned
        remove(list of strings) - names of columns to be removed from
                                the dataframe
        OUTPUTS:
        new (pandas dataframe) - cleaned dataframe
        """
        # narrow down to desired categories
        initial_size = df.shape
        print('Removing specified categories...')
        if remove:
            new = df.drop(remove, axis=1)
        else:
            new = df[['body', 'majority_type', 'ann_1', 'ann_2', 'ann_3']]
        #deal with nulls in desired columns
        new = self._replace_nulls(new)
        # drop remaining nulls
        print("Dropping remaining nulls: ")
        new = new.dropna(axis=0, how='any')
        # handle some messy columns:
        for col in ['ups', 'downs', 'created_utc']:
            if col in new.columns:
                new[col] = new[col].astype(int)
        # drop or consolidate where annotators disagreed
        if self._make_consensus:
            new = self._find_agreement(new)
        else:
            new = new[~new['majority_type'].isnull()]
        final_size = new.shape
        print("Down to {} of {} initial features".format(final_size[1], initial_size[1]))
        print("Removed {} rows".format(initial_size[0]-final_size[0]))
        return new

    def run(self, df, docs, labels='majority_type', cols_to_drop=None, direct_to_model=False):
        """
        Composite function. Cleans the data, engineers desired features, and
        splits into X and y components if desired. Also removes 'other' labels
        and redistributes 'None' labels into their majority annotation (the
        majority of votes cast in ann_1, ann_2 and ann_3 columns)
        INPUTS:
        df(pandas dataframe) - dataframe to be processed
        docs(string) - name of column containing text samples
        labels(string) - name of column containing labels (y-values)
        cols_to_drop (list of strings) - names of columns to be removed from
                                        consideration
        direct_to_model(bool) - boolean of whether to split data into X and y
                                components
        OUTPUTS:
        df(pandas df) - preprocessed dataframe (only when direct_to_model==False)
        X(pandas df) - preprocessed dataframe containing all features (only when
                    direct_to_model==True)
        y - preprocessed dataframe containing labels (only when direct_to_model
            ==True)
        """
        # feature engineering steps
        df = self.basic_feature_engineering(df)
        # custom step: remove 'other', remaining 'None's from categories
        df = df[~df[labels].isin(['other', 'None'])]
        #remove excess columns
        less = self.prepare_data(df, remove=cols_to_drop)
        # once more, just in case
        less = less.dropna()
        documents = less[docs]
        less = less.drop([docs], axis=1)

        # run it through tfidf
        print("Vectorizing comment bodies...")
        vectors, vocabulary = self.vectorize(documents)
        #shrink resulting tfidf_vectors
        print("Reducing tfidf vector dimensions...")
        U, S = self.matrix_reduction(vectors, power_level=1)
        print("Reconstructing DataFrame...")
        df_U = pd.DataFrame(U)
        seq = xrange(min(len(less), len(df_U)))
        less['fixit'], df_U['fixit'] = seq, seq
        df = pd.merge(less, df_U, on='fixit')
        if 'fixit' in df.columns:
            df = df.drop(['fixit'], axis=1)
        #debugging
        print("Checking reconstructed dataframe:")
        for c in df.columns:
            n = sum(df[c].isnull())
            L, C = [], []
            if n>0:
                L.append(n)
                C.append(c)
        if L:
            print("{} nulls in reconstructed df".format(len(L)))
            print("Cols affected: {}".format(C))
        else:
            print("No nulls found.")

        if direct_to_model:
            y = df[labels]
            X = df.drop([labels], axis=1)
            return X,y
        return df

    def vectorize(self, documents, max_features=None):
        """ Fits TfidfVectorizer on supplied corpus of documents, returns an
        array of tfidf vectors in the shape of n_documents x n_features, where
        n_features is <= max_features unless max_features is None, in which case
        n_features is dependent on the corpus size.
        INPUTS:
        documents(1 dimensional array) - corpus of documents, each of which is
                                        a snippet of text
        max_features (int) - the maximum number of features allowed in the
                            tfidf matrix; None unless stated
        OUTPUTS:
        vectors(sparse matrix) - Tf-idf weighted document term matrix of size
                                [n_samples, n_features]
        """
        if not self.tfidf:
            print("Creating initial TfidfVectorizer fit...")
            self.tfidf = TfidfVectorizer(stop_words='english', \
            tokenizer=self._tokenize, \
            max_features=None).fit(documents)
        vectors = self.tfidf.transform(documents).toarray()
        words = self.tfidf.get_feature_names
        return vectors, words

    def basic_feature_engineering(self, df):
        """Redistributes 'None' labels to the majority vote. Room for future
        functionality for different features
        INPUTS:
        df (pandas dataframe) - dataframe to undergo feature engineering
        OUTPUTS:
        df (pandas dataframe) - dataframe with new and/or improved features
        """
        # translate None labels into majority opinion if possible:
        print("Redistributing 'None' labels...")
        df = self.fix_None_labels(df)
        return df

    def fix_None_labels(self, df):
        """
        For all 'None' labels, checks the votes of the three annotators,
        and changes 'None' to the label chosen by the majority (no change in
        the case of ties)
        INPUTS:
        df (pandas dataframe) - dataframe with to be fixed
        OUTPUTS:
        df (pandas dataframe) - dataframe with None's changed to other labels
                                where possible
        """
        df['majority_type'] = df.apply(lambda row: self._none_fix_helper(row), axis=1)
        return df

    # ----------- personal utility and debugging below this line ------------ #
    def _check_df(self, df, verbose=False, name="df"):
        """
        Debugging function.
        For ensuring those pesky nulls and Nans have been dealth with.
        """
        #for dealing with bugs surrounding nulls
        print("Beginning _check_df of {} for nulls...".format(name))
        L= []
        if verbose:
            for c in df.columns:
                n = df[c].isnull().sum()
                if n:
                    print("{} has {} nulls".format(c, n))
                    L.append(n)
        else:
            for c in df.columns:
                n = df[c].isnull().sum()
                if n:
                    L.append(n)
        print("{} nulls in {} cols".format(sum(L), len(L)))

    def _find_desired_power(self, Sq, power_level):
        """
        Helper function used for additional matrix reduction if specified
        """
        for i in xrange(len(Sq)):
            if sum(Sq[:i]) >= power_level:
                return i
        else:
            return len(Sq)

    def _none_fix_helper(self, row):
        """
        Helper function in redistributing nulls.
        """
        cts = row[['ann_1', 'ann_2', 'ann_3']].value_counts()
        if row['majority_type'] == 'None' and cts.max() >1:
            x = row[['ann_1', 'ann_2', 'ann_3']].value_counts().idxmax()
        else:
            x = row['majority_type']
        return x

    def _replace_nulls(self, df):
        """
        Helper function. Used for handling nulls in specific ways depending on
        the column (feature) they appear in.
        """
        print('Replacing nulls in the following categories...')
        if 'gilded' in df.columns:
            print('gilded: changing strings to nums, all to True/False')
            df['gilded'].replace(to_replace=['0','1','2'], value=[0,1,2], inplace=True)
            df['gilded'] = df['gilded'] > 0
        elif 'post_depth' in df.columns:
            print('post_depth: changing nulls to 0\'s')
            df['post_depth'] = df['post_depth'].fillna(value=0)
            nulls = sum(df['post_depth'].isnull())
            print( "nulls in post_depth:{}".format(nulls))
        elif 'user_reports' in df.columns:
            print('user_reports: replacing odd values, filling nulls with 0\'s')
            df['user_reports'] = df['user_reports'].replace(to_replace='[]', value=1)
            df['user_reports'] = df['user_reports'].fillna(value=0)
        else:
            print("...none selected")
        return df

    def _test_run(self, df, docs, cols_to_drop=None, sep=True, reduce_step=True):
        """
        Debugging function. Much more verbose and flexible version of self.run.
        """
        # basic feature engineering steps
        df = self.basic_feature_engineering(df)
        #remove excess columns
        less = self.prepare_data(df, remove=cols_to_drop)
        # once more, just in case
        less = less.dropna()
        documents = less[docs]
        less = less.drop([docs], axis=1)
        print("Initial check of prepared df:")
        self._check_df(less)

        # run it through tfidf
        print("Vectorizing comment bodies...")
        vectors, vocabulary = self.vectorize(documents)

        #shrink resulting tfidf_vectors
        if reduce_step:
            print("Reducing tfidf vector dimensions...")
            U, S = self.matrix_reduction(vectors, power_level=None)
            print("Reconstructing DataFrame...")
            df_U = pd.DataFrame(U)
            print("Shapes: {}, {}".format(df_U.shape, less.shape))
            seq = xrange(min(len(df_U), len(less)))
            df_U['fixit'], less['fixit'] = seq, seq
            df = pd.concat([d.reset_index() for d in [less, df_U]], axis=1, ignore_index=True)
            if 'fixit' in df.columns:
                df = df.drop(['fixit'], axis=1)
            print("Done.\n")
            if sep:
                y = df['majority_type']
                X = df.drop(['majority_type'], axis=1)
                return X,y
            return less, df_U, df

        else:
            return vectors, None, None

    def _tokenize(self, text, type=None):
        """
        Custom tokenizer. Breaks on spaces and '/'s. Uses WordNetLemmatizer to
        lemmatize.
        """
        tokenizer = RegexpTokenizer(r"\s+|/", gaps=True)
        tokens = tokenizer.tokenize(text)
        return tokens

        stemmer = WordNetLemmatizer()
        stemmed = [stemmer.lemmatize(word) for word in tokens]
        return stemmed



if __name__ == '__main__':
    # load data
    train = pd.read_csv('data/train.csv')
    #columns to be removed
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                    'distinguished', 'edited', 'gilded', 'in_reply_to', 'is_first_post', \
                    'link_id', 'link_id_ann', 'majority_link', 'name',  \
                    'parent_id', 'replies', 'retrieved_on', 'saved', \
                    'score_hidden', 'subreddit', 'title', 'user_reports', \
                    'ann_1', 'ann_2', 'ann_3']

    # segment for testing
    subset = train[:1000]
    before = subset.majority_type.value_counts()
    P = PreProcessor()
    x,y = P.run(subset, 'body', cols_to_drop=remove_most)
