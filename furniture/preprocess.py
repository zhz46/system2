import json
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from gensim import corpora, models

import text_clean


def data_load(files='../dat/data/18000*.json'):
    # read all furniture jsons
    data = []
    features = ['products', 'parentProducts', 'brand', 'price_hint',
                'title', 'category_id', 'group_id', 'id', 'image_url', 'sku_id']
    for file_name in glob(files):
        with open(file_name) as f:
            temp = json.load(f)
            for sku in temp[:]:
                # read parent product list and product list
                parent_prod = set(sku['elasticsearch_result']['parentProducts'].keys())
                prod = set(sku['elasticsearch_result']['products'].keys())
                image = sku['main_image']['raw']
                sku['image_url'] = image
                if parent_prod:
                    sku['parentProducts'] = ' and '.join(sorted(parent_prod))
                    sku['products'] = ' and '.join(sorted(prod - parent_prod))
                else:
                    sku['parentProducts'] = np.nan
                    if prod:
                        sku['products'] = ' and '.join(sorted(prod))
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
    features = ['products', 'parentProducts', 'brand', 'price_hint',
                'title', 'category_id', 'group_id', 'id', 'sku_id', 'image_url']
    df = raw_df[features].copy()
    df = df.sort_values(by='category_id').reset_index(drop=True)

    # convert to float
    df.price_hint = df.price_hint.astype(float)
    # fill missing
    df.price_hint.fillna(df.price_hint.median(), inplace=True)
    return df


def image_merge(df, image_input):
    # read all image jsons
    data = []
    features = ['id', 'prelogits']
    for file_name in glob(image_input):
        with open(file_name) as f:
            temp = json.load(f)
            for sku in temp:
                for key in list(sku.keys()):
                    if key not in features:
                        del sku[key]
            data = data + temp

    # convert to df
    df_image = pd.DataFrame(data)

    # inner join text df and image df
    df = pd.merge(df_image, df, on='id', how='inner')
    return df


def product_map(file="../dat/pp_map.txt"):
    pp_map = {}
    with open(file) as f:
        for line in f:
            tokens = line.split('=>')
            pp_map[tokens[0].strip()] = tokens[1].strip()
    return pp_map

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
    text = text_clean.analyze(text)
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
def text_process(text, model):
    # text cleanup
    text = text_clean.analyze(text)
    # load stop_words
    stop_words = stopwords.words('english')
    # tokens filtered out stopwords
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
              if word not in stop_words and word in model.wv.vocab]
    return tokens


# filter out vectors without words in corpus
def df_filter(df, model):
    # return titles array
    titles = df.title.values
    # return processed titles bag of words
    docs = [text_process(title, model) for title in titles]
    filter_list = [any([word in model.wv.vocab for word in doc]) for doc in docs]
    df = df.loc[filter_list].copy().reset_index(drop=True)
    return df


# centroid doc2vec representation
def doc2vec_centroid(doc, model):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv.vocab]
    return np.mean(model.wv[doc], axis=0)


# tf-idf weighted word vectors
def doc2vec_tfidf(docs, model):
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    title_mat = np.zeros((len(corpus_tfidf), 300))
    for i in range(len(corpus_tfidf)):
        score_sum = np.sum(corpus_tfidf[i], axis=0)[1]
        title_mat[i, :] = np.sum([model.wv[dictionary[id]] * score / score_sum for id, score in corpus_tfidf[i]], axis=0)
    return title_mat


def product2vec(products, model):
    raw_list = list(products)
    product_mat = np.zeros((len(products), 300))
    for i in range(len(raw_list)):
        if pd.isnull(raw_list[i]):
            product_mat[i, :] = np.random.uniform(-0.25, 0.25, 300)
        else:
            tokens = re.split(' and |\s', raw_list[i])
            tokens = [token for token in tokens if token in model.wv.vocab]
            if not tokens:
                product_mat[i, :] = np.random.uniform(-0.25, 0.25, 300)
            else:
                product_mat[i, :] = np.mean(model.wv[tokens], axis=0)
    return product_mat


# from sklearn.decomposition import PCA
#
#
# # 51576208551 is total number words in the pretrained corpus
# def get_word_frequency(word, model):
#     return model.vocab[word].count/51576208551
#
# def idf(word, dictionary):
#     return np.log(dictionary.num_docs/dictionary.dfs[dictionary.token2id[word]])
#
# # A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# # convert a list of sentence with word2vec items into a set of sentence vectors
# def doc_to_vec(docs, model, algo, pca, embedding_size=300, a=0.001):
#     dictionary = corpora.Dictionary(docs)
#     doc_set = []
#     for doc in docs:
#         doc = [word for word in doc if word in model.vocab]
#         vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
#         doc_length = len(doc)
#         word_vectors = model.wv[doc]
#         for i in np.arange(doc_length):
#             if algo == "weight":
#                 a_value = a / (a + get_word_frequency(doc[i], model))  # smooth inverse frequency, SIF
#             elif algo == "tfidf":
#                 a_value = idf(doc[i], dictionary) # idf
#             else:
#                 raise ValueError('Please use either weight or tfidf for algo')
#             vs = np.add(vs, np.multiply(a_value, word_vectors[i]))  # vs += sif * word_vector
#         vs = np.divide(vs, doc_length)  # weighted average
#         doc_set.append(vs)  # add to our existing re-calculated set of sentences
#
#     if pca == 0:
#         return np.array(doc_set)
#
#     # calculate PCA of this doc set
#     pca = PCA(n_components=embedding_size)
#     pca.fit(np.array(doc_set))
#     u = pca.components_[0]  # the PCA vector
#     u = np.multiply(u, np.transpose(u))  # u x uT
#
#     # pad the vector?  (occurs if we have less sentences than embeddings_size)
#     if len(u) < embedding_size:
#         for i in range(embedding_size - len(u)):
#             u = np.append(u, 0)  # add needed extension for multiplication below
#
#     # resulting sentence vectors, vs = vs -u x uT x vs
#     doc_vecs = []
#     for vs in doc_set:
#         sub = np.multiply(u, vs)
#         doc_vecs.append(np.subtract(vs, sub))
#
#     return np.array(doc_vecs)