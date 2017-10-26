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
import matplotlib.pyplot as plt



class Processor(object):

    def __init__(self):
        self._make_consensus = False
        self.stemmers = {'WordNetLemmatizer': WordNetLemmatizer,
                         'SnowballStemmer': SnowballStemmer,
                         'PorterStemmer': PorterStemmer}

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
        vectorizer = TfidfVectorizer(stop_words='english', \
                     tokenizer=self._tokenize, max_features=None )
        vectors = vectorizer.fit_transform(documents).toarray()
        words = vectorizer.get_feature_names
        return vectors, words

    def _replace_nulls(self, df):
        print('Replacing nulls in the following categories...')
        if 'gilded' in df.columns:
            print('gilded: changing strings to nums, all to True/False')
            df['gilded'] = df['gilded'].replace(['0'], 0)
            df['gilded'] = df['gilded'] > 0
            df['gilded'].fillna(value=0, inplace=True)
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
        #deal with nulls in desired columns
        df = self._replace_nulls(new)
        # drop remaining nulls
        print("Dropping remaining nulls:\n{}".format(df.isnull().sum()))
        df = df.dropna(axis=0)
        # drop or consolidate where annotators disagreed
        if self._make_consensus:
            df = self._find_agreement(df)
        else:
            df = df[~df['majority_type'].isnull()]
        final_size = df.shape
        print("Down to {} of {} initial features".format(final_size[1], initial_size[1]))
        print("Removed {} rows".format(final_size[0]-initial_size[0]))
        return df

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
        if power_level>1:
            print "No. Power level can't be more than 1. Setting power_level to 0.9"
            power_level=0.9
        if n_comp >= matrix.shape[1] or n_comp<0:
            print "n_comp must be between 2 and the original number of features. Setting to half of original features"
            n_comp = max(2, matrix.shape[1]/2)
        #decompose matrix into U,S,V components
        SVD = TruncatedSVD(n_components=n_comp, n_iter=10).fit(matrix)
        U_trunc = SVD.transform(matrix)
        Sigma_sq = SVD.explained_variance_ratio_
        #shrink matrix to latent features that account for power_level fraction of total power
        if power_level:
            i = self._find_desired_power(Sigma_sq, power_level)
            U_trunc = U[:, :i]
        return U_trunc

    def _find_desired_power(self, Sq, power_level):
        for i in xrange(len(Sq)):
            if sum(Sq[:i]) >= power_level:
                return i
        else:
            return len(Sq)

    def run(self, df, docs, cols_to_drop=None, y=None):
        #remove excess columns
        less = self.prepare_data(df, remove=cols_to_drop)
        # run it through tfidf
        print("Vectorizing comment bodies...")
        vectors, vocabulary = self.vectorize(less[docs])
        initial_size = vectors.shape
        #shrink resulting tfidf_vectors
        print("Reducing tfidf vector dimentions...")
        U = self.matrix_reduction(vectors, power_level=0.99)
        final_size = U.shape
        print("...shrunk from {} features to {} features".format(initial_size[1], final_size[1]))
        # test
        print("Reconstructing DataFrame...")
        df_U = pd.DataFrame(U)
        df = pd.concat([less.drop('body', axis=1), df_U], axis=1)
        print("Done.")
        return df


def graph_latent_feature_power(sigma_matrix):
    power = sigma_matrix ** 2
    y = np.cumsum(power)/np.sum(power)
    X = xrange(len(y))
    plt.scatter(X,y)
    plt.show()


def test():
    subset = train[:5000]
    print("Beginning test on {} rows".format(subset.shape[0]))
    P = Processor()
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited', 'in_reply_to', 'is_first_post', \
                   'link_id', 'link_id_ann', 'majority_link', 'name',  \
                   'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    df = P.run(subset, 'body', cols_to_drop=remove_most)
    return df




if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    subset = train[:10000]



    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                    'distinguished', 'edited', 'in_reply_to', 'is_first_post', \
                    'link_id', 'link_id_ann', 'majority_link', 'name',  \
                    'parent_id', 'replies', 'retrieved_on', 'saved', \
                    'score_hidden', 'subreddit', 'title', 'user_reports', \
                    'ann_1', 'ann_2', 'ann_3']

    t = test()
    t.shape
    t.head()





    processor = Processor()
    content = processor.prepare_data(train)


    X_train, X_test, y_train, y_test = train_test_split(content['body'], content['majority_type'], test_size=0.25)

    X_train, vocab = processor.vectorize(X_train)


    GNB = GaussianNB().fit(X_train, y_train)
    MNB = MultinomialNB().fit(X_train, y_train)
    GNB_scores = cross_val_score(GNB, X_train, y=y_train, cv=5, n_jobs=-1)
    MNB_scores = cross_val_score(MNB, X_train, y=y_train, cv=5, n_jobs=-1)
    GNB_scores, MNB_scores

    y_train.unique()
