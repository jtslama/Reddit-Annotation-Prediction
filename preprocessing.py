import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class Processor(object):

    def __init__(self):
        self._make_consensus = False
        self.stemmers = {'WordNetLemmatizer': WordNetLemmatizer,
                         'SnowballStemmer': SnowballStemmer,
                         'PorterStemmer': PorterStemmer}
        self.SVD = None
        self.tfidf = None

    def _tokenize(self, text, type=None):
        #TODO check to find kind that handles spaces, formatting well
        tokenizer = RegexpTokenizer(r"\s+|/", gaps=True)
        tokens = tokenizer.tokenize(text)
        return tokens

        stemmer = WordNetLemmatizer()
        stemmed = [stemmer.lemmatize(word) for word in tokens]
        return stemmed

    def _find_agreement(self, df):
        #TODO finish function to fill in 'None's
        needs_consensus = True if df['majority_type']=='None' else False
        consensus = list(df['ann_1'], df['ann_2'], df['ann_3'])
        pass

    def vectorize(self, documents, max_features=None):
        if not self.tfidf:
            print("Creating initial TfidfVectorizer fit...")
            self.tfidf = TfidfVectorizer(stop_words='english', \
            tokenizer=self._tokenize, \
            max_features=None).fit(documents)
        vectors = self.tfidf.transform(documents).toarray()
        words = self.tfidf.get_feature_names
        return vectors, words

    def _replace_nulls(self, df):
        print('Replacing nulls in the following categories...')
        if 'gilded' in df.columns:
            print('gilded: changing strings to nums, all to True/False')
            df['gilded'] = df['gilded'].replace(['0'], 0)
            df['gilded'] = df['gilded'] > 0
            df['gilded'] = df['gilded'].fillna(value=0, inplace=True)
        elif 'post_depth' in df.columns:
            print('post_depth: changing nulls to 0\'s')
            df['post_depth'] = df['post_depth'].fillna(value=0)
            nulls = sum(df[col].isnull())
            print( "nulls in post_depth:{}".format(nulls))
        elif 'user_reports' in df.columns:
            print('user_reports: replacing odd values, filling nulls with 0\'s')
            df['user_reports'] = df['user_reports'].replace(to_replace='[]', value=1)
            df['user_reports'] = df['user_reports'].fillna(value=0)
        return df

    def prepare_data(self, df, remove=None):
        # narrow down to desired categories
        initial_size = df.shape
        print('Removing specified categories...\n{}'.format(remove))
        if remove:
            new = df.drop(remove, axis=1)
        else:
            new = df[['body', 'majority_type', 'ann_1', 'ann_2', 'ann_3']]
        # handle some messy columns:
        for col in ['ups', 'downs', 'created_utc']:
            if col in new.columns:
                new[col] = new[col].astype(int)
        #deal with nulls in desired columns
        new = self._replace_nulls(new)
        # drop remaining nulls
        print("Dropping remaining {} nulls:".format(new.isnull().sum()))
        new = new.dropna(axis=0, how='any')
        # drop or consolidate where annotators disagreed
        if self._make_consensus:
            new = self._find_agreement(new)
        else:
            new = new[~new['majority_type'].isnull()]
        final_size = new.shape
        print("Down to {} of {} initial features".format(final_size[1], initial_size[1]))
        print("Removed {} rows".format(final_size[0]-initial_size[0]))

        return new

    def feature_engineering(self, df):
        #use parent_ids to create a column
        relations = df.to_dict
        pass

    def matrix_reduction(self, matrix, n_comp=1000, power_level=None):
        """
        INPUTS:
        matrix(numpy array) - the matrix to be reduced
        n_comp (int) - dimension to initially reduce to, must be less than number of features (matrix.shape[1])
        power_level(float) - the percentage of explained variance desired (b/t 0 and 1)
        OUTPUTS:
        U_trunc - the trunkated user weight array (rows: items, columns: latent features, values: weights)
        Sigma_trunc - the trunkated power array (rows: latent features, columns: power, values: power)
        VT_trunc - the truncated features array (rows: item features, columns: latent features, values: weights)
        """
        initial_size = matrix.shape
        if power_level>1:
            print "No. Power level can't be more than 1. Setting power_level to 0.9"
            power_level=0.9
        if n_comp >= matrix.shape[1] or n_comp<0:
            print "n_comp must be between 2 and the original number of features. Setting to half of original features"
            n_comp = max(2, matrix.shape[1]/2)
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

    def _find_desired_power(self, Sq, power_level):
        for i in xrange(len(Sq)):
            if sum(Sq[:i]) >= power_level:
                return i
        else:
            return len(Sq)

    def run(self, df, docs, cols_to_drop=None, sep=True):
        #remove excess columns
        less = self.prepare_data(df, remove=cols_to_drop)
        print type(less)
        documents = df[docs]
        less = less.drop([docs], axis=1)
        print type(less)
        # return less

        # run it through tfidf
        print("Vectorizing comment bodies...")
        vectors, vocabulary = self.vectorize(documents)
        #shrink resulting tfidf_vectors
        print("Reducing tfidf vector dimensions...")
        U, S = self.matrix_reduction(vectors, power_level=None)
        print("Reconstructing DataFrame...")
        df_U = pd.DataFrame(U)
        #debugging:
        print("u_dim, less_dim: {}, {}".format(df_U.shape, less.shape))
        for df in [df_U, less]:
            print("starting df null check...")
            L=[]
            for c in df.columns:
                n = sum(df[c].isnull())
                if n:
                    L.append(n)
            if L:
                print("{} nulls".format(sum(L)))
            else:
                print "...None found"

        df = pd.concat([less, df_U], axis=1)
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
            print("Cols affected:\n{}".format(C))

        print("Done.\n")
        y = df['majority_type']
        X = df.drop(['majority_type'], axis=1)
        if sep:
            return X,y
        # return df
        return less

    def full_run(self, df, docs, cols_to_drop=None, y=None):
        """ in case you want to run the whole thing at once
        """
        #remove excess columns
        less = self.prepare_data(df, remove=cols_to_drop)
        # splitting (to avoid information leakage during tfidf)
        y = less['majority_type']
        X = less.drop(['majority_type'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
      # run it through tfidf
        print("Vectorizing comment bodies...")
        train_vectors, train_vocab = self.vectorize(df_train[docs])
        test_vectors, test_vocab = self.vectorize(df_train[docs])
        #shrink resulting tfidf_vectors
        print("Reducing tfidf vector dimentions...")
        U_train, S_train = self.matrix_reduction(train_vectors, power_level=0.95)
        U_test, S_test = self.matrix_reduction(test_vectors, power_level=0.95)
        # test
        print("Reconstructing DataFrame...")
        df_U_train, df_U_test = pd.DataFrame(U_train), pd.DataFrame(U_test)
        df_train = pd.concat([df_train.drop('body', axis=1), df_U_train], axis=1)
        df_test = pd.concat([df_test.drop('body', axis=1), df_U_test], axis=1)

        print("Done.\n")

        return df_train, df_test

def _test(size=2000):
    subset = train[:size]
    df_train, df_test = train_test_split(subset, test_size=0.25)
    print("Beginning test on {} rows".format(subset.shape[0]))
    P = Processor()
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited', 'in_reply_to', 'is_first_post', \
                   'link_id', 'link_id_ann', 'majority_link', 'name',  \
                   'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    # X,y
    less = P.run(df_train, 'body', cols_to_drop=remove_most, sep=True)


    # return X,y
    return less


def graph_latent_feature_power(sigma_matrix):
    power = sigma_matrix ** 2
    y = np.cumsum(power)/np.sum(power)
    X = xrange(len(y))
    plt.plot((0,len(y)),(0.9,0.9), 'r-')
    plt.plot((0,len(y)),(0.95,0.95), 'g-')
    plt.plot((0,len(y)),(0.99,0.99), 'b-')
    plt.scatter(X,y)
    # plt.savefig('power_explained.png')
    plt.show()





if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    subset = train[:2000]
    train.name



    #ISSUE IS WITH LESS: IT COMES OUT WITH LOTS OF NULLS
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                    'distinguished', 'edited', 'in_reply_to', 'is_first_post', \
                    'link_id', 'link_id_ann', 'majority_link', 'name',  \
                    'parent_id', 'replies', 'retrieved_on', 'saved', \
                    'score_hidden', 'subreddit', 'title', 'user_reports', \
                    'ann_1', 'ann_2', 'ann_3']

    less = _test(2000)
    type(less)
    less = less.drop(['majority_type'], axis=1)
    less.info()

    P = Processor()
    test_less = P.prepare_data(subset, remove_most)
    test_less.info()

    for c in less:
        print less[c].isnull().sum()
    comps = [dfu, less]

    for df in [less]:
        print("starting df...")
        L, C = [], []
        for c in df.columns:
            n = sum(df[c].isnull())
            if sum(x[c].isnull()):
                print c, sum(x[c].isnull())
        else:
            print 'Fine'





    s = subset.iloc[3, 4]
    t = P._tokenize(s)
    t
    # testing for new feature (parent_id's annotation)
    # problem: of ~3k parent_ids (/5k), only ~150 are in id's
    sum(subset.in_reply_to == subset.parent_id)
    subset.parent_id

    d = defaultdict(str)
    for i, a in zip(subset.name, subset.majority_type):
        d[i]=a
    test = subset.parent_id.apply(lambda x: d[x] if x in d else 'No_Parent')


    test.value_counts()
    for n in range(1,7):
        z = subset[subset.parent_id.str.startswith('t{}_'.format(n))]['parent_id']
        print len(z)


    z = subset[subset.parent_id.str.startswith('t3_')]['parent_id']
    z.head()
    len(z)
    subset.post_depth.value_counts()
    p_ids, ids = set(subset.parent_id), set(subset.name)
    len(set(z))
    len(p_ids.intersection(ids))


    # graph_latent_feature_power(S)
    """
    For graphing purposes
    power = pd.read_csv('/Users/jt/Desktop/ex_var_ratios.csv', header=None)
    %pwd
    graph_latent_feature_power(power)
    """
