import json
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import utils


def data_load(files='../../desktop/new_data/*.json'):
    # read all furniture jsons
    data = []
    features = ['products', 'parentProducts', 'brand', 'price_hint',
                'title', 'category_id', 'group_id', 'id', 'image_url']
    for file_name in glob(files):
        with open(file_name) as f:
            temp = json.load(f)
            for sku in temp:
                # read parent product list and product list
                parent_prod = list(sku['elasticsearch_result']['parentProducts'].keys())
                prod = list(sku['elasticsearch_result']['products'].keys())
                image = sku['main_image']['raw']
                sku['image_url'] = image
                if parent_prod:
                    sku['parentProducts'] = parent_prod[0]
                    if prod[0] == parent_prod[0]:
                        sku['products'] = prod[1]
                    else:
                        sku['products'] = prod[0]
                else:
                    sku['parentProducts'] = np.nan
                    if prod:
                        sku['products'] = prod[0]
                    else:
                        sku['products'] = np.nan
                for key in list(sku.keys()):
                    if key not in features:
                        del sku[key]
            data = data + temp
    return data


def pre_process(raw_data):
    # convert list into dataframe
    raw_df = pd.DataFrame(raw_data)
    # for group with size 1, mark them as no_group skus
    single_group_df = raw_df.groupby('group_id').filter(lambda x: len(x) <= 1)
    raw_df.loc[single_group_df.index.values, 'group_id'] = np.nan
    # divide no_group skus, primary skus and secondary skus
    # no_group_df = raw_df.loc[raw_df.group_id.isnull()]
    # group_df = raw_df.loc[raw_df.group_id.notnull()]
    # primary_df = group_df.groupby('group_id').apply(lambda x: x.iloc[0])
    # second_df = group_df.groupby('group_id').apply(lambda x: x.iloc[1:])
    # # concat above components in a better order for later
    # raw_df = pd.concat([primary_df, no_group_df, second_df]).reset_index()

    # save index for numpy array
    raw_df['original_index'] = raw_df.index

    # select key features
    # features = ['title', 'category_id', 'category_level_0', 'category_level_1',
    #             'brand', 'attributes', 'price_hint', 'description', 'sku_id']
    features = ['products', 'parentProducts', 'brand', 'price_hint',
                'title', 'category_id', 'group_id', 'id', 'original_index']
    fts = {}
    for i in range(len(features)):
        fts[features[i]] = i
    df = raw_df[features].copy()

    # pre-process data
    # convert to float
    df.price_hint = df.price_hint.astype(float)
    # fill missing
    df.price_hint.fillna(df.price_hint.median(), inplace=True)
    # drop two skus that do not have 'meaningful' titles
    # df = df.drop(df.index[[76774, 113749]])
    return (df, fts)


# def map_generate(df):
#     # map secondary sku index to its primary sku index
#     second2primary = {}
#     # map primary sku index to its group index list
#     primary_neighbor = {}
#     # map group_id to its group index list
#     group_map = df.groupby('group_id').groups
#     for values in group_map.values():
#         primary_neighbor[values[0]] = values
#         for i in range(1, len(values)):
#             if values[i] not in second2primary:
#                 second2primary[values[i]] = values[0]
#     return second2primary, primary_neighbor

# tokenize and stem function for feature extraction
def tokenize_and_stem(text):
    # load nltk's stemmer object
    stemmer = SnowballStemmer("english")
    # text cleanup
    text = utils.analyze(text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # filtered_tokens = []
    # for token in tokens:
    #     if re.search('[a-zA-Z]', token):
    #         filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in tokens]
    return stems


# seq = range(200, 601, 50)
# var_track = []
# for i in seq:
#     svd = TruncatedSVD(n_components=i, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
#     tfidf_rd = svd.fit_transform(tfidf_matrix)
#     var_track.append(svd.explained_variance_ratio_.sum())


# title processing for tfidf
def title_process(titles):
    # calculate tfidf matrix for title
    tf = TfidfVectorizer(analyzer='word', min_df=0, max_df=0.9, tokenizer=tokenize_and_stem, stop_words='english')
    tfidf_matrix = tf.fit_transform(titles)
    # Latent semantic analysis and re-normalization for tfidf matrix (dimension reduction)
    svd = TruncatedSVD(n_components=300, algorithm='arpack', n_iter=5, random_state=None, tol=0.0)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    tfidf_rd = lsa.fit_transform(tfidf_matrix)
    return tfidf_rd


# tokenize and stem function for feature extraction
def text_process(text):
    # text cleanup
    text = utils.analyze(text)
    # load stop_words
    stop_words = stopwords.words('english')
    # tokens filtered out stopwords
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stop_words]
    return tokens


# filter out vectors without words in corpus
def df_filter(df, wv):
    # return titles array
    titles = df.title.values
    # return processed titles bag of words
    docs = [text_process(title) for title in titles]
    docs = np.array(docs)
    filter_list = [any([word in wv.vocab for word in doc]) for doc in docs]
    docs = docs[filter_list]
    df = df.loc[filter_list].copy().reset_index(drop=True)
    df.original_index = df.index
    return df, docs


# centroid doc2vec representation
def doc2vec_centroid(doc, wv):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in wv.vocab]
    return np.mean(wv[doc], axis=0)