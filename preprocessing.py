import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np




def _tokenize(text):
    #TODO check to find kind that handles spaces, formatting well
    tokenizer = RegexpTokenizer("\s+", gaps=True)
    tokens = tokenizer.tokenize(text)
    return tokens






class Processor(object):

    def __init__(self):
        self._make_consensus = False
        self.stemmers = {'WordNetLemmatizer': WordNetLemmatizer,
                         'SnowballStemmer': SnowballStemmer,
                         'PorterStemmer': PorterStemmer}

    def _tokenize(self, text, type=None):
        #TODO check to find kind that handles spaces, formatting well
        tokenizer = RegexpTokenizer(r"\w+|\s+", gaps=True)
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

    def prepare_data(self, df):
        # narrow down to desired categories
        new = df[['body', 'majority_type', 'ann_1', 'ann_2', 'ann_3']]
        # drop nulls in body and label columns
        narrowed = new[~new['body'].isnull() & ~new['majority_type'].isnull()]
        # drop or consolidate where annotators disagreed
        if self._make_consensus:
            narrowed = self._find_agreement(narrowed)
        else:
            narrowed = narrowed[~narrowed['majority_type'].isnull()]

        return narrowed




if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    basics = isolate_basics(train)

    processor = Processor()
    content = processor.prepare_data(train)
    X, vocab = processor.vectorize(content.body)
