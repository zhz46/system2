import sqlite3
import numpy as np
import pandas as pd
import json
import time
from glob import glob
from multiprocessing import Pool
from sklearn.preprocessing import normalize

from preprocess import data_load, pre_process, title_process, text_process, doc2vec_centroid, product_map
from distance import title_only, image_only


def weight_series(length, ini=0.95):
    weights = []
    for i in range(length):
        weights.append(ini**i)
    return np.array(weights)


def co_view_select(sqlite_file = '../dat/rexagm.db'):
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


# select co_view data from sqlite database
co_view = co_view_select()

# load raw_data
raw_data = data_load()

# pre_process data
df, fts = pre_process(raw_data)


##############################################
# tf-idf score
# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

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

# return titles array
titles = new_df.title.values

# process unstructured titles
title_mat = title_process(titles)

# combine title and other features
mat = np.concatenate((new_df.values.copy(), title_mat), axis=1)

# divide mats
candidate_mat = mat[:len(primary_df)+len(no_group_df)].copy()
second_mat = mat[len(primary_df)+len(no_group_df):].copy()


# generate top-k recommendation list given an index
def score(idx, dist=image_only, candidate_mat=candidate_mat, second_mat=second_mat, fts=fts,
          category_map=category_map, co_view_map = co_view_map):

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
    return result


tfidf_score = parallel(score, list(co_view_map.keys()))
tfidf_score = np.array(tfidf_score)
ratio = np.sum(tfidf_score, axis=0)[0]/np.sum(tfidf_score, axis=0)[1]



###########################################
# image score
# read all image jsons
files = '../../desktop/raw/18000*.json'
data = []
features = ['id', 'prelogits']
for file_name in glob(files):
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

# expand prelogits into its own dataframe
df_prelogit = new_df.prelogits.apply(pd.Series)
prelogit_mat = normalize(df_prelogit.values)

features = ['sku_id', 'group_id', 'category_id']
# build feature index
fts = {}
for i in range(len(features)):
    fts[features[i]] = i

# need to make a copy before concatenate!!, 20 times faster
index_mat = new_df[features].values.copy()

# concat original
mat = np.concatenate((index_mat, prelogit_mat), axis=1)

# divide mats
candidate_mat = mat[:len(primary_df)+len(no_group_df)].copy()
second_mat = mat[len(primary_df)+len(no_group_df):].copy()

# generate image scores
start = time.time()
image_score = parallel(score, list(co_view_map.keys()))
print(time.time() - start)
image_score = np.array(image_score)
ratio = np.sum(image_score, axis=0)[0]/np.sum(image_score, axis=0)[1]
print(ratio)