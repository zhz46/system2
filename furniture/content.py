import json
import nltk
import re
import time
import utils
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem.snowball import SnowballStemmer
from numpy.random import randint




# read all furniture jsons
data = []
for file_name in glob('../../desktop/data/*.json'):
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
        data = data + temp


# convert list into dataframe
raw_df = pd.DataFrame(data)


# select key features
# features = ['title', 'category_id', 'category_level_0', 'category_level_1',
#             'brand', 'attributes', 'price_hint', 'description', 'sku_id']
features = ['products', 'parentProducts', 'brand', 'price_hint', 'title', 'category_id', 'id']
fts = {}
for i in range(len(features)):
    fts[features[i]] = i
df = raw_df[features].copy()


# pre-process data
# convert to float
df.price_hint = df.price_hint.astype(float)
# fill missing
df.price_hint.fillna(df.price_hint.median(), inplace=True)


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


tf = TfidfVectorizer(analyzer='word', min_df=0, max_df=0.9, tokenizer=tokenize_and_stem, stop_words='english')
tfidf_matrix = tf.fit_transform(df['title'])


# seq = range(200, 601, 50)
# var_track = []
# for i in seq:
#     svd = TruncatedSVD(n_components=i, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
#     tfidf_rd = svd.fit_transform(tfidf_matrix)
#     var_track.append(svd.explained_variance_ratio_.sum())

# title processing
def title_process(df):
    # calculate tfidf matrix for title
    tf = TfidfVectorizer(analyzer='word', min_df=0, max_df=0.9, tokenizer=tokenize_and_stem, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['title'])
    # Latent semantic analysis and re-normalization for tfidf matrix (dimension reduction)
    svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    tfidf_rd = lsa.fit_transform(tfidf_matrix)
    return tfidf_rd


title_mat = title_process(df)
# col_name = ['t'+str(i) for i in range(200)]
# df_clean = pd.concat([df, pd.DataFrame(title_mat, columns=col_name)], axis=1)
mat = np.concatenate((df.values, title_mat), axis=1)


# product processing using pandas df
# def prod_process(a, b):
#     # calculate products and parentProducts distance
#     # either one is missing
#     if pd.isnull(a.loc['products']) or pd.isnull(b.loc['products']):
#         prod_dist = 0.8
#     # same products
#     elif a.loc['products'] == b.loc['products']:
#         prod_dist = 0
#     # one's product is same as another's parentProduct
#     elif a.loc['products'] == b.loc['parentProducts'] or b.loc['products'] == a.loc['parentProducts']:
#         prod_dist = 0.1
#     # one's parentProduct is part of another's product
#     elif (pd.notnull(a.loc['parentProducts']) and a.loc['parentProducts'] in b.loc['products']) or \
#             (pd.notnull(b.loc['parentProducts']) and b.loc['parentProducts'] in a.loc['products']):
#         prod_dist = 0.3
#     # one's product is part of another's product
#     elif a.loc['products'] in b.loc['products'] or b.loc['products'] in a.loc['products']:
#         prod_dist = 0.3
#     else:
#         prod_dist = 1
#     return prod_dist


def prod_process(a, b):
    # calculate products and parentProducts distance
    # either one is missing
    if pd.isnull(a[fts['products']]) or pd.isnull(b[fts['products']]):
        prod_dist = 0.8
    # same products
    elif a[fts['products']] == b[fts['products']]:
        prod_dist = 0
    # one's product is same as another's parentProduct
    elif a[fts['products']] == b[fts['parentProducts']] or b[fts['products']] == a[fts['parentProducts']]:
        prod_dist = 0.1
    # one's parentProduct is part of another's product
    elif (pd.notnull(a[fts['parentProducts']]) and a[fts['parentProducts']] in b[fts['products']]) or \
            (pd.notnull(b[fts['parentProducts']]) and b[fts['parentProducts']] in a[fts['products']]):
        prod_dist = 0.3
    # one's product is part of another's product
    elif a[fts['products']] in b[fts['products']] or b[fts['products']] in a[fts['products']]:
        prod_dist = 0.3
    else:
        prod_dist = 1
    return prod_dist


# price processing
def price_process(a, b):
    i = a[fts['price_hint']]
    j = b[fts['price_hint']]
    return np.abs(i - j)/(i + j)


# brand processing
def brand_process(a, b):
    if a[fts['brand']] == b[fts['brand']]:
        return 0
    return 1


# calculate weighted distance
def mixed_dist(a, b, prod_wt=0.5, brand_wt=0.2, title_wt=0.2, price_wt=0.1):
    # calculate title_dist
    title_dist = title_only(a, b)
    # calculate prod_dist
    prod_dist = prod_process(a, b)
    # calculate price_dist
    price_dist = price_process(a, b)
    # calculate brand_dist
    brand_dist = brand_process(a, b)
    distance = np.dot([prod_wt, brand_wt, title_wt, price_wt],
                  [prod_dist, brand_dist, title_dist, price_dist])
    return distance


# calculate title distance only
def title_only(a, b):
    title_dist = (1 - np.dot(a[7:], b[7:])) * 0.5
    return title_dist


# generate top-k recommendation list given an index
def query(ind, k=30, dist=mixed_dist, data=mat):
    idx = data[ind, fts['id']]
    # indices = np.arange(len(data))
    # temp_data = data[indices!=ind, :]
    # temp_data = np.delete(data, ind, 0)
    # dist_mat = np.apply_along_axis(dist, axis=1, arr=temp_data, b=data[ind, :])
    dist_mat = np.apply_along_axis(dist, axis=1, arr=data, b=data[ind, :])
    temp_ind = np.argpartition(dist_mat, k+1)[:k+1]
    top_ind = temp_ind[np.argsort(dist_mat[temp_ind])]
    if ind in top_ind:
        top_ind = top_ind[top_ind != ind]
    else:
        top_ind = top_ind[:k]
    top_score = dist_mat[top_ind]
    # top_idy = temp_data[top_ind, fts['id']]
    top_idy = data[top_ind, fts['id']]
    result = []
    for idy, value in zip(top_idy, top_score):
        pair = dict()
        pair['idX'] = idx
        pair['idY'] = idy
        pair['method'] = 'content_based_v1'
        pair['score'] = value
        result.append(pair)
    return result


# generate top-k recommendation list given an index
def query(ind, k=30, dist=mixed_dist, data=mat):
    idx = data[ind, fts['id']]
    cate_id = data[ind, fts['category_id']]
    temp_data = data[data[:, fts['category_id']]==cate_id]
    # indices = np.arange(len(data))
    # temp_data = data[indices!=ind, :]
    # temp_data = np.delete(data, ind, 0)
    dist_mat = np.apply_along_axis(dist, axis=1, arr=temp_data, b=data[ind, :])
    # dist_mat = np.apply_along_axis(dist, axis=1, arr=data, b=data[ind, :])
    temp_ind = np.argpartition(dist_mat, k+1)[:k+1]
    top_ind = temp_ind[np.argsort(dist_mat[temp_ind])]
    if ind in top_ind:
        top_ind = top_ind[top_ind != ind]
    else:
        top_ind = top_ind[:k]
    top_score = dist_mat[top_ind]
    top_idy = temp_data[top_ind, fts['id']]
    # top_idy = data[top_ind, fts['id']]
    result = []
    for idy, value in zip(top_idy, top_score):
        pair = dict()
        pair['idX'] = idx
        pair['idY'] = idy
        pair['method'] = 'content_based_v1'
        pair['score'] = value
        result.append(pair)
    return result


# def pair_out(inds):
#     pairs = []
#     for ind in inds:
#         pairs.extend(query(ind))
#     return pairs


# random sample indices
sample = randint(0, len(data), 1000)


# parallel computing
def parallel(func, chunk=sample, p=6):
    pool = Pool(processes=p)
    result = pool.map(func, chunk)
    flat_result = [sku for rs_list in result for sku in rs_list]
    return flat_result


# generate overall recommendation pair list
start = time.time()
rs_output = parallel(query, sample, 6)
print(time.time() - start)

# output content_rs.json
with open('content_rs.json', 'w') as f:
    json.dump(rs_output, f)


# with open('content_rs.json') as f:
#     zz = json.load(f)


# output data.json
output = ['id', 'brand', 'image_url', 'title', 'category_id']
df_output = raw_df[output]
df_output = df_output.rename(columns={'category_id':'node_id'})
df_output.to_json('data.json', orient='records')


# def test(id, k=30, dist=mixed_dist, data=mat):
#     temp_data = np.delete(data, id, 0)
#     dist_mat = np.apply_along_axis(dist, axis=1, arr=temp_data, b=mat[id, :])
#     idx = np.argpartition(dist_mat, k)[:k]
#     top_idx = idx[np.argsort(dist_mat[idx])]
#     result = temp_data[top_idx, :]
#     return result


# with open('data.json') as f:
#     zz = json.load(f)

def test(ind, data=mat):
    idx = data[ind, fts['id']]
    cate_id = data[ind, fts['category_id']]
    temp_data = data[data[:, fts['category_id']]==cate_id]

sample = randint(0, len(data), 6)
pool = Pool(processes=6)
sample [1,100,1000,20000]
%timeit test(0)
%timeit pool.map(test, sample)

%timeit query(0)
%timeit pool.map_async(query, sample).get()
%timeit pool.map(query, sample)

sample = randint(0, len(data), 600)
%timeit parallel(query, sample, 6)