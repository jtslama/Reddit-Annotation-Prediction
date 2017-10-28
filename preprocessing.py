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


class PreProcessor(object):
    """Primarily for cleaning the data and converting the comment bodies to tfidf
    vectors
    """
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

    def prepare_data(self, df, remove=None):
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
        print("Removed {} rows".format(final_size[0]-initial_size[0]))
        return new

    def basic_feature_engineering(self, df):
        # translate None labels into majority opinion if possible:
        print("Redistributing 'None' labels...")
        df = self.fix_None_labels(df)

        return df

    def matrix_reduction(self, matrix, n_comp=2000, power_level=None):
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
        if n_comp >= matrix.shape[1]:
            print "n_comp must be between 2 and the original number of features. Setting to half of original features"
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


    def _none_fix_helper(self, row):
        cts = row[['ann_1', 'ann_2', 'ann_3']].value_counts()
        if row['majority_type'] == 'None' and cts.max() >1:
            x = row[['ann_1', 'ann_2', 'ann_3']].value_counts().idxmax()
        else:
            x = row['majority_type']
        return x

    def fix_None_labels(self, df):

        df['majority_type'] = df.apply(lambda row: self._none_fix_helper(row), axis=1)

        """
        subset.head()
        for i in subset.index:
            if subset.loc[i, 'majority_type'] == 'None':
                subset.loc[i, 'majority_type'] = subset.loc[i, ['ann_1', 'ann_2', 'ann_3']].value_counts().idxmax()

        subset.majority_type.value_counts()
        """
        return df


    def _find_desired_power(self, Sq, power_level):
        for i in xrange(len(Sq)):
            if sum(Sq[:i]) >= power_level:
                return i
        else:
            return len(Sq)

    def run(self, df, docs, labels='majority_type', cols_to_drop=None, direct_to_model=False):
        # feature engineering steps
        df = self.basic_feature_engineering(df)
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
        U, S = self.matrix_reduction(vectors, power_level=None)
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


    def _test_run(self, df, docs, cols_to_drop=None, sep=True, reduce_step=True):
        #For debugging purposes
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
            df = df.drop(['fixit'], axis=1)
            print("Done.\n")
            if sep:
                y = df['majority_type']
                X = df.drop(['majority_type'], axis=1)
                return X,y
            return less, df_U, df

        else:
            return vectors, None, None


    def _check_df(self, df, verbose=False, name="df"):
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


def _test(size=10000):
    subset = train[:size]
    df_train, df_test = train_test_split(subset, test_size=0.25)
    print("\nBeginning test on {} rows".format(subset.shape[0]))
    P = PreProcessor()
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                   'distinguished', 'edited', 'gilded', 'in_reply_to', 'is_first_post', \
                   'link_id', 'link_id_ann', 'majority_link', 'name',  \
                   'parent_id', 'replies', 'retrieved_on', 'saved', \
                   'score_hidden', 'subreddit', 'title', 'user_reports', \
                   'ann_1', 'ann_2', 'ann_3']

    X,y = P.run(df_train, 'body', cols_to_drop=remove_most)

    return X,y



# def graph_latent_feature_power(sigma_matrix, save_as=None):
#     power = sigma_matrix ** 2
#     y = np.cumsum(power)/np.sum(power)
#     X = xrange(len(y))
#     plt.figure(figsize=(20,20))
#     plt.plot((0,len(y)),(0.9,0.9), 'r-')
#     plt.plot((0,len(y)),(0.95,0.95), 'g-')
#     plt.plot((0,len(y)),(0.99,0.99), 'b-')
#     plt.scatter(X,y)
#     if save_as:
#         plt.savefig('{}.png'.format(save_as))
#     plt.show()

def graph_label_distribs(series, save_as=None):
    total = series.sum()
    d = dict(series)
    ordered = sorted(zip(d.values(), d.keys()))
    heights, labels = zip(*ordered)
    heights = [h*1.0/total for h in heights]
    positions = xrange(len(heights))

    plt.figure(figsize=(10,10))
    plt.bar(positions, heights, alpha=0.5)
    plt.xticks(positions, labels, rotation=45)
    plt.ylabel('Proportion of Comments')
    plt.title('Comment Types')
    if save_as:
        plt.savefig('{}.png'.format(save_as))
    plt.show()


if __name__ == '__main__':
    # load data
    train = pd.read_csv('data/train.csv')
    subset = train[:1000]
    remove_most = ['Unnamed: 0', 'annotations', 'archived', 'author', 'date', \
                    'distinguished', 'edited', 'gilded', 'in_reply_to', 'is_first_post', \
                    'link_id', 'link_id_ann', 'majority_link', 'name',  \
                    'parent_id', 'replies', 'retrieved_on', 'saved', \
                    'score_hidden', 'subreddit', 'title', 'user_reports', \
                    'ann_1', 'ann_2', 'ann_3']
    #test everything
    X,y = _test(size=1000)

    # test none_fixer
    before = subset.majority_type.value_counts()
    P = PreProcessor()
    x,y = P.run(subset, 'body', cols_to_drop=remove_most)
    after = y.value_counts()
    before
    after
    x_fix, y_fix = P.run(train, 'body', cols_to_drop=remove_most)
    y_fix.value_counts()
    graph_label_distribs(y_fix.value_counts())


    # testing for new feature (parent_id's annotation)
    # problem: of ~3k parent_ids (/5k), only ~150 are in id's
    subset = train[:50000].drop()
    sum(subset.in_reply_to == subset.parent_id)
    subset.parent_id
    #248 of parent_id start with t3 (t3=posts, not comments)
    len(subset[subset.parent_id.str.startswith('t3_')]['parent_id'])

    d = defaultdict(str)
    ct = 0
    for i in subset.parent_id:
        if i in subset.name:
            ct += 1
    ct
    for i, a in zip(subset.name, subset.majority_type):
        d[i]=a
    test = subset.parent_id.apply(lambda x: d[x] if x in d else 'No_Parent')
    test.value_counts()
    for n in range(1,7):
        z = subset[subset.parent_id.str.startswith('t{}_'.format(n))]['parent_id']
        print len(z)



    z.head()
    len(z)
    subset.post_depth.value_counts()
    p_ids, ids = set(subset.parent_id), set(subset.name)
    len(set(z))
    len(p_ids.intersection(ids))



    """
    # For graphing purposes
    power = pd.read_csv('/Users/jt/Desktop/ex_var_ratios.csv', header=None)
    %pwd
    graph_latent_feature_power(power)
    """
