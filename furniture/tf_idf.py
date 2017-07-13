import json
import nltk
import time
import utils
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem.snowball import SnowballStemmer
from preprocess import data_load, pre_process, map_generate
from distance import mixed_dist


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


# title processing
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


# load raw_data
raw_data = data_load()

# pre_process data
df, fts = pre_process(raw_data)

# return titles array
titles = df.title.values

# process unstructured titles
title_mat = title_process(titles)

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
with open('tfidf_rs.json', 'w') as f:
    json.dump(rs_output, f)


# output data.json
# output = ['id', 'brand', 'image_url', 'title', 'category_id']
# df_output = raw_df[output]
# df_output = df_output.rename(columns={'category_id':'node_id'})
# df_output.to_json('data.json', orient='records')



# if __name__ == "__main__":
#     main()

