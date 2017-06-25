import json
import nltk
import re
import numpy as np
import pandas as pd
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


# read all furniture jsons
data = []
for file_name in glob('../../desktop/data/*.json'):
    with open(file_name) as f:
        temp = json.load(f)
        data = data + temp


# convert list into dataframe
raw_df = pd.DataFrame(data)


# select key features
# features = ['title', 'category_id', 'category_level_0', 'category_level_1',
#             'brand', 'attributes', 'price_hint', 'description', 'sku_id']
features = ['brand', 'price_hint', 'title', 'category_id']
df = raw_df[features]



# title processing
# load nltk's English stopwords
stopwords = nltk.corpus.stopwords.words('english')


# load nltk's stemmer object
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# tokenize and stem function for feature extraction
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# text = '78 in. High 3-Wide 3-Tier Premium Knock-Down Locker in Cream (36 in. W x 18 in. D x 78 in. H)'
# tokenize_and_stem(text)
# # use tf-idf to extract features
# def tfidf_extract(data):
#     tf = TfidfVectorizer(analyzer='word', min_df=0, max_df=0.9, tokenizer=tokenize_and_stem, stop_words='english')
#     tfidf_matrix = tf.fit_transform(data['title'])
#     feature_names = tf.get_feature_names()
#     sku_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
#     sku_df = sku_df.drop('null', axis=1)
#     sku_df['sku_id'] = data['sku_id']
#     sku_df = sku_df.dropna()
#     return sku_df

# calculate tfidf matrix for title
tf = TfidfVectorizer(analyzer='word', min_df=0, max_df=0.9, tokenizer=tokenize_and_stem, stop_words='english')
tfidf_matrix = tf.fit_transform(df['title'])


# Latent semantic analysis and re-normalization for tfidf matrix (dimension reduction)
seq = range(200, 1001, 50)
var_track = []
for i in seq:
    svd = TruncatedSVD(n_components=i, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
    tfidf_rd = svd.fit_transform(tfidf_matrix)
    var_track.append(svd.explained_variance_ratio_.sum())

svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
tfidf_rd = lsa.fit_transform(tfidf_matrix)



from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=20 + 1, algorithm='auto').fit(tfidf_rd)

# from numpy.random import randint
# sample = randint(0, len(tfidf_rd), 1000)


def query(id, k):
    neighbor = nbrs.kneighbors(tfidf_rd[id, :],return_distance=False)
    return df.ix[neighbor[id], :]

rcd = rs(25, 1000)


# def query(id, rcd_set=rcd):
#     return df.ix[rcd_set[id], :]
#
# query(0)


# from sklearn.metrics.pairwise import pairwise_distances
# pw_mat = pairwise_distances(tfidf_rd, metric='cosine')
#
# from scipy.spatial.distance import pdist
# pw_mat = pdist(tfidf_rd, metric="cosine")


