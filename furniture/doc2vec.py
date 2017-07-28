import json
import time
import numpy as np
from sklearn.preprocessing import normalize
from gensim import corpora, models
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from preprocess import data_load, pre_process, df_filter, doc2vec_centroid
from distance import mixed_dist
from tools import query, parallel
from doc2vec_weight import doc_to_vec


input = '../dat/data/18000*.json'
output1 = '../output/doc2vec_centroid.json'
output2 = '../output/doc2vec_weight.json'
trained_model = '../../Desktop/trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec'
method_name = 'content_based_v3.2'

# load raw_data
raw_data = data_load(input)

# pre_process data
df, fts = pre_process(raw_data)

# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format(trained_model)

# filter out empty bags of word
df, docs = df_filter(df, model_ft)

# words mean representation of docs
title_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
# title_mat = normalize(doc_to_vec(docs=docs, model=model_ft, algo='weight', pca=1))
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
with open(output1, 'w') as f:
    json.dump(rs_output, f)


# if __name__ == "__main__":
#     main()