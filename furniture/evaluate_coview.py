import sqlite3
import numpy as np
import pandas as pd
import json
import pickle
import time
from glob import glob
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import Doc2Vec

from preprocess import data_load, pre_process, title_process, df_filter, text_process, doc2vec_centroid, product_map, doc2vec_tfidf
from doc2vec_weight import doc_to_vec
from distance import title_only, image_only


co_view_query = '../dat/co_view_query.pkl'
text_input = '../dat/data/18000*.json'
image_input = '../dat/raw/18000*.json'
# text_input = '/srv/zz/temp/18000*.json'
# image_input = '/yg/analytics/rex/tensorflow/image2vec/dat/output/raw/18000*.json'
word2vec_model = '../trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_3_wordNgrams_3_ws_5.vec'
doc2vec_model = '../trained_models/dm.model'

def weight_series(length, ini=0.95):
    weights = []
    for i in range(length):
        weights.append(ini**i)
    return np.array(weights)


def co_view_select(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    # # store table column names
    # c.execute(
    #     "SELECT * FROM dIIRtCk LIMIT 5"
    #     )
    # co_view = c.fetchall()
    # names = [description[0] for description in c.description]
    # query furniture co-view data
    c.execute(
        "select skuY, skuX, ggXY "
        "from dIIRtCk "
        "where nodeY like '18000%' and nodeX like '18000%' "
        "and nodeX = nodeY "
        "order by skuX "
        )
    co_view = c.fetchall()
    return co_view


def generate_map(df, co_view):
    # secondonary map to primary
    id_list = df.sku_id.values
    sec2pri = {}
    group_map = df.groupby('group_id').groups
    for values in group_map.values():
        for i in range(1, len(values)):
            sec2pri[id_list[values[i]]] = id_list[values[0]]

    # build a co-view map for each sku => top co-view score skus
    co_view_map = {}
    sku_ids = set(id_list)
    for pair in co_view:
        if pair[2] is None:  # no co-view score
            continue
        if pair[1] not in sku_ids or pair[0] not in sku_ids:  # sku_id not exist
            continue
        idx = pair[1]
        if idx not in co_view_map:
            co_view_map[idx] = {}
        if pair[0] in sec2pri:
            idy = sec2pri[pair[0]]
        else:
            idy = pair[0]
        if idy in co_view_map[idx]:
            if pair[2] > co_view_map[idx][idy]:
                co_view_map[idx][idy] = pair[2]
            else:
                continue
        else:
            co_view_map[idx][idy] = pair[2]

    # replace co_view score by ranked score
    for idx in co_view_map:
        sort_arr = sorted(co_view_map[idx], key=lambda x : x[1], reverse=True)
        weight_arr = weight_series(len(sort_arr))
        for i in range(len(sort_arr)):
            co_view_map[idx][sort_arr[i]] = weight_arr[i]

    return co_view_map, sec2pri


def df_sort(df):
    # divide df to 3 parts
    no_group_df = df.loc[df.group_id.isnull()]
    group_df = df.loc[df.group_id.notnull()]
    primary_df = group_df.groupby('group_id').apply(lambda x: x.iloc[0])
    second_df = group_df.groupby('group_id').apply(lambda x: x.iloc[1:])

    # generate candidate df for calculation
    candidate_df = pd.concat([primary_df, no_group_df]).reset_index(drop=True)
    candidate_df = candidate_df.sort_values(by='category_id').reset_index(drop=True)

    # build map
    category_map = candidate_df.groupby('category_id').groups
    for key, value in category_map.items():
        value = value[0], value[-1] + 1
        category_map[key] = value

    # stack everything for title process
    new_df = pd.concat([candidate_df, second_df]).reset_index(drop=True)
    return new_df, category_map


def combo_dist(a, b, fts, wt):
    title_dist = (1 - np.dot(a[len(fts): len(fts)+300], b[len(fts): len(fts)+300])) * 0.5
    image_dist = 1 - np.dot(a[(len(fts)+300):], b[(len(fts)+300):])
    return wt * title_dist + (1 - wt) * image_dist


# calculate importance score for each sku
def score(idx, dist, candidate_mat, second_mat, fts, category_map, co_view_map, wt):
    # get co_view dict for the query sku
    co_view_dict = co_view_map[idx]
    # get query data
    if idx in sec2pri:
        cal_data = second_mat[second_mat[:,fts['sku_id']] == idx]
    else:
        cal_data = candidate_mat[candidate_mat[:, fts['sku_id']]== idx]
    cal_data = cal_data.reshape(cal_data.shape[1],)
    cate_id = cal_data[fts['category_id']]
    cate_index = category_map[cate_id]
    # get candidate data, with same category_id and not itself
    candidate_data = candidate_mat[cate_index[0]:cate_index[1]]
    # calculate similarity for each candidate
    if dist == 'combo_dist':
        dist_mat = np.apply_along_axis(combo_dist, axis=1, arr=candidate_data, b=cal_data, fts=fts, wt=wt)
    else:
        dist_mat = np.apply_along_axis(dist, axis=1, arr=candidate_data, b=cal_data, fts=fts)

    # sort candidate data
    index_arr = np.argsort(dist_mat)
    candidate_sort = candidate_data[index_arr]
    scores = weight_series(len(candidate_sort))

    # find idy's score
    weight_sum = 0
    full_sum = 0
    for idy, weight_y in list(co_view_dict.items()):
        score_y = scores[candidate_sort[:, fts['sku_id']] == idy]
        if not score_y:
            continue
        weight_sum += weight_y * score_y[0]
        full_sum += weight_y ** 2
    return weight_sum, full_sum


# parallel computing
def parallel(func, chunk, p=6):
    pool = Pool(processes=p)
    result = pool.map_async(func, chunk).get()
    pool.close()
    return result

# co_view = co_view_select(sqlite_file)
# with open('../dat/co_view_query.pkl', 'wb') as output:
#     pickle.dump(co_view, output, pickle.HIGHEST_PROTOCOL)

with open(co_view_query, 'rb') as f:
    co_view = pickle.load(f)

# load raw_data
raw_data = data_load(text_input)

# pre_process data
df, fts = pre_process(raw_data)

##############################################
# tf-idf score
# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

new_df, category_map = df_sort(df)

# return titles array
titles = new_df.title.values

# process unstructured titles
title_mat = title_process(titles)

# combine title and other features
mat = np.concatenate((new_df.values.copy(), title_mat), axis=1)

# divide mats
candidate_mat = mat[:(len(df) - len(sec2pri))].copy()
second_mat = mat[(len(df) - len(sec2pri)):].copy()


# make a wrapper for score function
def score_text(idx):
    return score(idx, dist=title_only, candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
          category_map=category_map, co_view_map = co_view_map, wt=1)


tfidf_score = parallel(score_text, list(co_view_map.keys()))
tfidf_score = np.array(tfidf_score)
tfidf_ratio = np.sum(tfidf_score, axis=0)[0]/np.sum(tfidf_score, axis=0)[1]
print(tfidf_ratio)

###########################################
# doc2vec centroid
# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format(word2vec_model)

# filter out empty bags of word
df, _ = df_filter(df, model_ft)

# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

new_df, category_map = df_sort(df)

titles = new_df.title.values
docs = [text_process(title, model_ft) for title in titles]
docs = np.array(docs)

# words mean representation of docs
# title_mat = normalize(doc2vec_tfidf(docs, model_ft))
title_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
# title_mat = normalize(doc_to_vec(docs=docs, model=model_ft, algo='tfidf', pca=1))
mat = np.concatenate((new_df.values.copy(), title_mat), axis=1)

# divide mats
candidate_mat = mat[:(len(df) - len(sec2pri))].copy()
second_mat = mat[(len(df) - len(sec2pri)):].copy()


# make a wrapper for score function
def score_text(idx):
    return score(idx, dist=title_only, candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
          category_map=category_map, co_view_map=co_view_map, wt=1)

centroid_score = parallel(score_text, list(co_view_map.keys()))
centroid_score = np.array(centroid_score)
centroid_ratio = np.sum(centroid_score, axis=0)[0]/np.sum(centroid_score, axis=0)[1]

###########################################
# doc2vec dm
# load pretrained dm model
model_dm = Doc2Vec.load(doc2vec_model)

# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

new_df, category_map = df_sort(df)

# return titles array
titles = new_df.title.values
# return processed titles bag of words
docs = [text_process(title, model_dm) for title in titles]

# words mean representation of docs
title_mat = normalize(np.array([model_dm.infer_vector(doc) for doc in docs]))
mat = np.concatenate((new_df.values.copy(), title_mat), axis=1)

# divide mats
candidate_mat = mat[:(len(df) - len(sec2pri))].copy()
second_mat = mat[(len(df) - len(sec2pri)):].copy()

# make a wrapper for score function
def score_text(idx):
    return score(idx, dist=title_only, candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
          category_map=category_map, co_view_map=co_view_map)

dm_score = parallel(score_text, list(co_view_map.keys()))
dm_score = np.array(centroid_score)
dm_ratio = np.sum(centroid_score, axis=0)[0]/np.sum(centroid_score, axis=0)[1]

###########################################
# text + image score
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
del data

# inner join text df and image df
df = pd.merge(df_image, df, on='id', how='inner')

# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

new_df, category_map = df_sort(df)

# expand prelogits into its own dataframe
df_prelogit = new_df.prelogits.apply(pd.Series)
prelogit_mat = normalize(df_prelogit.values)

# return titles array
titles = new_df.title.values

# process unstructured titles
title_mat = title_process(titles)

features = ['sku_id', 'group_id', 'category_id']
# build feature index
fts = {}
for i in range(len(features)):
    fts[features[i]] = i

# need to make a copy before concatenate!!, 20 times faster
index_mat = new_df[features].values.copy()

# concat original
# mat = np.concatenate((index_mat, prelogit_mat), axis=1)
mat = np.concatenate((index_mat, title_mat, prelogit_mat), axis=1)

# divide mats
candidate_mat = mat[:(len(df) - len(sec2pri))].copy()
second_mat = mat[(len(df) - len(sec2pri)):].copy()


# # make a wrapper for score function
# def score_image(idx):
#     return score(idx, dist=image_only, candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
#           category_map=category_map, co_view_map=co_view_map, wt=1)
#
# # generate image scores
# image_score = parallel(score_image, list(co_view_map.keys()))
# image_score = np.array(image_score)
# image_ratio = np.sum(image_score, axis=0)[0]/np.sum(image_score, axis=0)[1]
# print(image_ratio)

# calculate importance score for each sku
for i in np.arange(0.1, 1, 0.05):
    start = time.time()
    def score_combo(idx):
        return score(idx, dist='combo_dist', candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
              category_map=category_map, co_view_map=co_view_map, wt=i)

    combo_score = parallel(score_combo, list(co_view_map.keys()))
    combo_score = np.array(combo_score)
    ratio = np.sum(combo_score, axis=0)[0] / np.sum(combo_score, axis=0)[1]
    print(ratio)
    print(time.time()-start)