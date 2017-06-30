import json
import nltk
import time
import utils
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
from numpy.random import randint
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors




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
# drop two skus that do not have 'meaningful' titles
df = df.drop(df.index[[81656, 156522]])




# tokenize and stem function for feature extraction
def text_process(text):
    # load nltk's stemmer object
    # stemmer = SnowballStemmer("english")
    # text cleanup
    text = utils.analyze(text)
    # load stop_words
    stop_words = stopwords.words('english')
    # tokens filtered out stopwords
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stop_words]
    # stems of words
    # stems = [stemmer.stem(t) for t in tokens]
    return tokens


# return titles array
titles = df.title.values
# return processed titles bag of words
docs = [text_process(title) for title in titles]
docs = np.array(docs)

# generate frequency table
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# generate titles dictionary, 27418 terms/words
# dictionary = corpora.Dictionary(docs)

# generate word counts/corpus sparse array
# corpus = [dictionary.doc2bow(text) for text in texts]


# from Google news, 100 billion words; the model has 300 dim vectors for 3 million words and phrases
# filename = 'GoogleNews-vectors-negative300.bin.gz'
# word2vec_google = KeyedVectors.load_word2vec_format(filename, binary=True)

# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format('../../Desktop/trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec')


# centroid doc2vec representation
def doc2vec_centroid(doc, wv):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in wv.vocab]
    return np.mean(wv[doc], axis=0)

# count=0
# for i in range(len(docs)):
#     new_doc = [word for word in docs[i] if word in model_ft.vocab]
#     if len(new_doc)==0:
#         print((i, docs[i]))



# words mean representation of docs
title_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
mat = np.concatenate((df.values, title_mat), axis=1)


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


# calculate title distance only
def title_only(a, b):
    title_dist = (1 - np.dot(a[7:], b[7:])) * 0.5
    return title_dist


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


# calculate mixed wm distance
def mixed_wm(a, b, doc_a, doc_b, prod_wt=0.5, brand_wt=0.2, title_wt=0.2, price_wt=0.1):
    # calculate title_dist
    title_dist = wm_dist(doc_a, doc_b)
    # calculate prod_dist
    prod_dist = prod_process(a, b)
    # calculate price_dist
    price_dist = price_process(a, b)
    # calculate brand_dist
    brand_dist = brand_process(a, b)
    distance = np.dot([prod_wt, brand_wt, title_wt, price_wt],
                  [prod_dist, brand_dist, title_dist, price_dist])
    return distance

from scipy.special import expit
def wm_dist(doc_a, doc_b, model=model_ft):
    out = model.wmdistance(doc_a, doc_b)/10
    if out > 1:
        print('doc_a')
        print('doc_b')
    return out

# result = []
# for i in range(len(df)):
#     dis = wm_dist(docs[0], docs[i])
#     result.append(dis)
# result

ori = df.values
# generate top-k recommendation list given an index
def query(ind, k=30, dist=mixed_wm, data=ori, docs=docs):
    # get id and category_id for given index
    idx = data[ind, fts['id']]
    cate_id = data[ind, fts['category_id']]
    # get candidate data and calculate distance
    indices = np.arange(len(data))
    temp_data = data[(data[:, fts['category_id']]==cate_id) & (indices!=ind)]
    if len(temp_data) == 0:
        return [{'idX': idx, 'idY': idx, 'score':1, 'method':'content_based_v1'}]
    if dist == mixed_dist:
        dist_mat = np.apply_along_axis(dist, axis=1, arr=temp_data, b=data[ind, :])
    else:
        temp_docs = docs[(data[:, fts['category_id']]==cate_id) & (indices!=ind)]
        dist_mat = [dist(data[ind, :], temp_data[i], docs[ind], temp_docs[i])
                    for i in np.arange(len(temp_data))]
        dist_mat = np.array(dist_mat)
    if len(temp_data) > k:
        temp_ind = np.argpartition(dist_mat, k)[:k]
        top_ind = temp_ind[np.argsort(dist_mat[temp_ind])]
    else:
        top_ind = np.argsort(dist_mat)

    top_score = dist_mat[top_ind]
    top_idy = temp_data[top_ind, fts['id']]
    result = []
    for idy, value in zip(top_idy, top_score):
        pair = dict()
        pair['idX'] = idx
        pair['idY'] = idy
        pair['method'] = 'content_based_v1'
        pair['score'] = 1 - value
        result.append(pair)
    return result


# query for centroid method
# parallel computing
def parallel(func, chunk, p=6):
    pool = Pool(processes=p)
    result = pool.map_async(func, chunk).get()
    flat_result = [sku for rs_list in result for sku in rs_list]
    return flat_result


sample = randint(0, len(df), 5000)

start = time.time()
rs_output = parallel(query, sample[:500], 6)
print(time.time()-start)


# output content_rs.json
with open('content_wm.json', 'w') as f:
    json.dump(rs_output, f)

query(0, k=10, dist=mixed_wm, data= df.values)




for i in rs_output:
    i['method'] = 'content_based_wm_v2.2'


with open('content_centroid.json') as f:
    zz= json.load(f)

for i in zz:
    i['method'] = 'content_based_centroid_v2.1'

with open('content_centroid.json') as f:
    zz = json.load(f)
