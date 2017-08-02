import json
import time
import numpy as np
from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec

from preprocess import data_load, pre_process, text_process
from distance import mixed_dist
from tools import query, parallel


input = '../dat/data/18000*.json'
output = '../output/doc2vec_dm.json'
trained_model = '../trained_models/dm.model'
trained_model_full = '../trained_models/dm.model_full'
method_name = 'content_based_v3.4'

# load raw_data
raw_data = data_load(input)

# pre_process data
df, fts = pre_process(raw_data)

# pre-trained model from gensim doc2vec
model_dm = Doc2Vec.load(trained_model_full)

# return titles array
titles = df.title.values
# return processed titles bag of words
docs = [text_process(title) for title in titles]

# words mean representation of docs
# title_mat = normalize(np.array([model_dm.infer_vector(doc) for doc in docs]))
title_mat = normalize(np.array([model_dm.docvecs['t_%s' % i] for i in range(len(docs))]))
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


# if __name__ == "__main__":
#     main()