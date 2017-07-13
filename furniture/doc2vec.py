import json
import nltk
import time
import utils
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from preprocess import data_load, pre_process, map_generate
from distance import mixed_dist
from doc2vec_weight import doc_to_vec


# tokenize and stem function for feature extraction
def text_process(text):
    # text cleanup
    text = utils.analyze(text)
    # load stop_words
    stop_words = stopwords.words('english')
    # tokens filtered out stopwords
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stop_words]
    # stems of words
    return tokens


# centroid doc2vec representation
def doc2vec_centroid(doc, wv):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in wv.vocab]
    return np.mean(wv[doc], axis=0)



# calculate mixed wm distance
# def mixed_wm(a, b, doc_a, doc_b, prod_wt=0.5, brand_wt=0.2, title_wt=0.2, price_wt=0.1):
#     # calculate title_dist
#     title_dist = wm_dist(doc_a, doc_b)
#     # calculate prod_dist
#     prod_dist = prod_process(a, b)
#     # calculate price_dist
#     price_dist = price_process(a, b)
#     # calculate brand_dist
#     brand_dist = brand_process(a, b)
#     distance = np.dot([prod_wt, brand_wt, title_wt, price_wt],
#                   [prod_dist, brand_dist, title_dist, price_dist])
#     return distance
#
# def wm_dist(doc_a, doc_b, model=model_ft):
#     out = model.wmdistance(doc_a, doc_b)/10
#     if out > 1:
#         print('doc_a')
#         print('doc_b')
#     return out

# load raw_data
raw_data = data_load()

# pre_process data
df, fts = pre_process(raw_data)

# return titles array
titles = df.title.values

# return processed titles bag of words
docs = [text_process(title) for title in titles]
docs = np.array(docs)

# generate titles dictionary, 27418 terms/words
# dictionary = corpora.Dictionary(docs)

# generate word counts/corpus sparse array
# corpus = [dictionary.doc2bow(text) for text in texts]

# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format(
    '../../Desktop/trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec')

# words mean representation of docs
# title_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
title_mat = normalize(doc_to_vec(docs=docs, model=model_ft, algo='weight', pca=1))
# combine title and other features
mat = np.concatenate((df.values, title_mat), axis=1)

# build two maps for query
second2primary, primary_neighbor = map_generate(df)


# generate top-k recommendation list given an index
def query(query_ind, k=30, dist=mixed_dist, data=mat, fts=fts,
          second2primary=second2primary, primary_neighbor=primary_neighbor):

    # set a switch for secondary sku to primary sku
    switch = 0
    # store the start index of secondary sku
    second_start = len(mat) - len(second2primary)

    if query_ind >= second_start:    # if query sku is a secondary sku
        ind = second2primary[query_ind]
        switch = 1
        if ind >= second_start:
            raise ValueError('Not a primary sku')
    else:                      # if query sku is a primary sku or non-group sku
        ind = query_ind

    cal_data = data[ind]
    cate_id = data[ind, fts['category_id']]
    # get candidate data and calculate distance
    temp_data = data[:second_start]
    indices = np.arange(len(temp_data))
    temp_data = temp_data[(temp_data[:, fts['category_id']] == cate_id) & (indices != ind)]
    if len(temp_data) == 0:
        return [{'idX': cal_data[fts['id']], 'idY': cal_data[fts['id']], 'score': 1, 'method': 'content_based_v2.1'}]
    dist_mat = np.apply_along_axis(dist, axis=1, arr=temp_data, a=cal_data, fts=fts)
    if len(temp_data) > k:
        temp_ind = np.argpartition(dist_mat, k)[:k]
        top_ind = temp_ind[np.argsort(dist_mat[temp_ind])]
    else:
        top_ind = np.argsort(dist_mat)
    top_score = dist_mat[top_ind]
    top_idy = temp_data[top_ind, fts['id']]

    # get original index of top skus
    top_ori_index = temp_data[top_ind, fts['original_index']]
    result = []
    for i in range(len(top_idy)):
        # if recommend sku is primary sku, calculate nearest one in the group
        if top_ori_index[i] < len(primary_neighbor):
            group_i = primary_neighbor[top_ori_index[i]]
            group_data = data[group_i]
            # if original query sku is secondary, then use original data to calculate
            if switch == 1:
                cal_data = data[query_ind]
            group_dist = np.apply_along_axis(dist, axis=1, arr=group_data, a=cal_data, fts=fts)
            best_ind = np.argmin(group_dist)
            value = group_dist[best_ind]
            idy = group_data[best_ind, fts['id']]
        else:
            idy = top_idy[i]
            value = top_score[i]
        pair = dict()
        pair['idX'] = data[query_ind, fts['id']]
        pair['idY'] = idy
        pair['method'] = 'content_based_v2.1'
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

# generate overall recommendation pair list
start = time.time()
rs_output = parallel(query, range(len(df)), 6)
print(time.time() - start)

# output content_rs.json
with open('doc2vec_weight.json', 'w') as f:
    json.dump(rs_output, f)



#
# if __name__ == "__main__":
#     main()


