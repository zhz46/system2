import json
import time
import numpy as np

from preprocess import data_load, pre_process, title_process
from tools import query, parallel
from distance import mixed_dist


input = '../dat/data/18000*.json'
output = '../output/tfidf_rs.json'
method_name = 'content_based_v3.1'

# load raw_data
raw_data = data_load(input)

# pre_process data
df, fts = pre_process(raw_data)

# return titles array
titles = df.title.values

# process unstructured titles
title_mat = title_process(titles)

# combine title and other features
mat = np.concatenate((df.values.copy(), title_mat), axis=1)

# build map
category_map = df.groupby('category_id').groups
for key, value in category_map.items():
    value = value[0], value[-1] + 1
    category_map[key] = value


# make a wrapper of query function
def query_wrapper(ind):
    return query(ind, k=30, dist=mixed_dist, data=mat, fts=fts, map=category_map, method_name=method_name)

# generate overall recommendation pair list
start = time.time()
rs_output = parallel(query_wrapper, range(len(df)), 6)
print(time.time() - start)

# output content_rs.json
with open(output, 'w') as f:
    json.dump(rs_output, f)


# output data.json
# output = ['id', 'brand', 'image_url', 'title', 'category_id']
# df_output = raw_df[output]
# df_output = df_output.rename(columns={'category_id':'node_id'})
# df_output.to_json('data.json', orient='records')


# if __name__ == "__main__":
#     main()