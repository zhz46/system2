import json
import time
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Pool
from queue import PriorityQueue
from sklearn.preprocessing import normalize

from preprocess import data_load, pre_process

input1 = '/srv/zz/temp/*.json'
input2 = '/yg/analytics/rex/tensorflow/image2vec/dat/output/raw/18000*.json'

# load raw_data
raw_data = data_load(input1)

# pre_process data
df_text, _ = pre_process(raw_data)

# read all image jsons
# files = '../../desktop/raw/18000*.json'
data = []
features = ['id', 'prelogits']
for file_name in glob(input2):
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
df = pd.merge(df_image, df_text, on='id', how='inner')
df = df.sort_values(by='category_id').reset_index(drop=False)

# expand prelogits into its own dataframe
df_prelogit = df.prelogits.apply(pd.Series)
prelogit_mat = normalize(df_prelogit.values)

features = ['id', 'group_id', 'category_id']
# need to make a copy before concatenate!!, 20 times faster
index_mat = df[features].values.copy()

# concat original
mat = np.concatenate((index_mat, prelogit_mat), axis=1)

# build feature index
fts = {}
for i in range(len(features)):
    fts[features[i]] = i

# build map
category_map = df.groupby('category_id').groups
for key, value in category_map.items():
    value = value[0], value[-1] + 1
    category_map[key] = value

def image_only(b, a, fts):
    image_dist = np.dot(a[len(fts):], b[len(fts):])
    # image_dist = np.linalg.norm(a[len(fts):] - b[len(fts):])
    return image_dist


def query(ind, k=30, dist=image_only, data=mat, fts=fts, map=category_map):
    # get query data
    cal_data = data[ind]
    cate_id = cal_data[fts['category_id']]
    cate_index = map[cate_id]
    # get candidate data, with same category_id and not itself
    candidate_data = data[cate_index[0]:cate_index[1]]

    # if no candidate data
    if len(candidate_data) == 1:
        return [{'idX': cal_data[fts['id']], 'idY': 'N/A', 'score': 'N/A', 'method': 'image_similarity_v1'}]

    # calculate similarity for each candidate
    sim_mat = np.apply_along_axis(dist, axis=1, arr=candidate_data, a=cal_data, fts=fts)
    # sim_mat = 1 - dist_mat
    candidate_id = candidate_data[:, fts['id']]
    candidate_gid = candidate_data[:, fts['group_id']]

    # use PQ to get top k recommendation
    pq = PriorityQueue()
    group_max = {}
    # put no_group sku into PQ and max group sku in map
    for i in range(len(sim_mat)):
        if candidate_id[i] == cal_data[fts['id']]:
            continue
        if pd.isnull(candidate_gid[i]):     # if no group_id
            pq.put((sim_mat[i], candidate_id[i]))
        elif candidate_gid[i] not in group_max:  # has a new group_id
            group_max[candidate_gid[i]] = (sim_mat[i], candidate_id[i])
        else:   # has a old group_id
            if group_max[candidate_gid[i]][0] < sim_mat[i]:
                group_max[candidate_gid[i]] = (sim_mat[i], candidate_id[i])
            else:
                continue
        if pq.qsize() > k:
            pq.get()
    # put max group sku in PQ
    for values in group_max.values():
        pq.put(values)
        if pq.qsize() > k:
            pq.get()
    # move recommendations out from PQ
    rs_size = pq.qsize()
    result = [0] * rs_size
    for i in range(rs_size-1, -1, -1):
        output = pq.get()
        pair = dict()
        pair['idX'] = cal_data[fts['id']]
        pair['idY'] = output[1]
        pair['method'] = 'image_similarity_v1'
        pair['score'] = output[0]
        result[i] = pair
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
with open('image_sim.json', 'w') as f:
    json.dump(rs_output, f)


# mat = np.concatenate((df[features].values.copy(), prelogit_mat), axis=1)
# mat2 = np.concatenate((df[features].values, prelogit_mat), axis=1)
