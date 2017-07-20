import json
import time
import numpy as np
import pandas as pd
from queue import PriorityQueue
from multiprocessing import Pool

from preprocess import data_load, pre_process, title_process
from distance import mixed_dist


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
# second2primary, primary_neighbor = map_generate(df)


# generate top-k recommendation list given an index
def query(ind, k=30, dist=mixed_dist, data=mat, fts=fts):
    # get query data
    cal_data = data[ind]
    cate_id = cal_data[fts['category_id']]
    # get candidate data, with same category_id and not itself
    indices = np.arange(len(data))
    candidate_data = data[(data[:, fts['category_id']] == cate_id) & (indices != ind)]

    # if no candidate data
    if len(candidate_data) == 0:
        return [{'idX': cal_data[fts['id']], 'idY': 'N/A', 'score': 'N/A', 'method': 'content_based_v3.1'}]

    # calculate similarity for each candidate
    dist_mat = np.apply_along_axis(dist, axis=1, arr=candidate_data, a=cal_data, fts=fts)
    sim_mat = 1 - dist_mat
    candidate_id = candidate_data[:, fts['id']]
    candidate_gid = candidate_data[:, fts['group_id']]

    # use PQ to get top k recommendation
    pq = PriorityQueue()
    group_max = {}
    # put no_group sku into PQ and max group sku in map
    for i in range(len(sim_mat)):
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
        pair['method'] = 'content_based_v3.1'
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
with open('tfidf_rs.json', 'w') as f:
    json.dump(rs_output, f)


# output data.json
# output = ['id', 'brand', 'image_url', 'title', 'category_id']
# df_output = raw_df[output]
# df_output = df_output.rename(columns={'category_id':'node_id'})
# df_output.to_json('data.json', orient='records')



# if __name__ == "__main__":
#     main()

