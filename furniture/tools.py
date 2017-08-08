import numpy as np
import pandas as pd
from multiprocessing import Pool
from queue import PriorityQueue
from distance import title_only, image_only, combo_dist


# generate top-k recommendation list given an index
def query(ind, k, data, fts, map, method_name, **kwargs):
    # get query data
    cal_data = data[ind]
    cate_id = cal_data[fts['category_id']]
    cate_index = map[cate_id]
    # get candidate data, with same category_id
    candidate_data = data[cate_index[0]:cate_index[1]]

    # if no candidate data, just itself
    if len(candidate_data) == 1:
        return [{'idX': cal_data[fts['id']], 'idY': 'N/A', 'score': 'N/A', 'method': method_name}]

    # calculate similarity for each candidate
    if 'wt' in kwargs:
        dist_mat = np.apply_along_axis(combo_dist, axis=1, arr=candidate_data, b=cal_data, fts=fts, wt=kwargs['wt'])
    else:
        dist_mat = np.apply_along_axis(title_only, axis=1, arr=candidate_data, b=cal_data, fts=fts)
    sim_mat = 1 - dist_mat
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
        pair['method'] = method_name
        pair['score'] = output[0]
        result[i] = pair
    return result


# parallel computing
def parallel(func, chunk, p=6):
    pool = Pool(processes=p)
    result = pool.map_async(func, chunk).get()
    pool.close()
    flat_result = [sku for rs_list in result for sku in rs_list]
    return flat_result