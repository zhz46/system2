import json
import numpy as np
from glob import glob
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import Doc2Vec

from preprocess import data_load, pre_process, title_process, text_process, doc2vec_centroid, product_map, doc2vec_tfidf
from evaluation import cluster_eval, class_eval
from doc2vec_weight import doc_to_vec


text_input = '../dat/data/18000*.json'
doc2vec_model = '../trained_models/dm.model_full'
word2vec_model = '../trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec'


# load raw_data
raw_data = data_load(text_input)

# pre_process data
df, fts = pre_process(raw_data)

# map products
pp_map = product_map()
df.products = df.products.map(lambda x: pp_map[x] if x in pp_map else x)

# use group size > 1624 for testing, 20 groups
df_test = df.dropna(subset=['products'])
df_test = df_test.groupby('products').filter(lambda x: len(x) > 1624)

# return titles array
titles = df_test.title.values


# 1. tfidf test
################
tfidf_mat = title_process(titles)
cluster_eval(tfidf_mat, df_test['products'].values, 20)

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(tfidf_mat, df_test['products'].values, C=i))

# 2. doc2vec test
################
# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format(word2vec_model)
# model_ft = Doc2Vec.load(word2vec_model)

# return processed titles bag of words
docs = [text_process(title, model_ft) for title in titles]
docs = np.array(docs)


# 2.1 centroid method
####################
# centroid_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
centroid_mat = normalize(doc2vec_tfidf(docs, model_ft))
cluster_eval(centroid_mat, df_test['products'].values, 20)

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(centroid_mat, df_test['products'].values, C=i))


# 2.2 weighted average method
######################
weight_mat = doc_to_vec(docs=docs, model=model_ft, algo='weight', pca=1)
weight_mat = normalize(weight_mat)
cluster_eval(weight_mat, df_test['products'].values, 20)

# result = []
# seq = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
# for i in seq:
#     weight_df = pd.DataFrame(doc_to_vec(docs=docs, model=model_ft, fun=get_word_frequency, a=i))
#     weight_df['products'] = df_test['products'].values
#     result.append(cluster_eval(weight_df, 70))

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(weight_mat, df_test['products'].values, C=i))

# 2.3 dm
########################
dm_mat = normalize(np.array([model_ft.docvecs['t_%s' % i] for i in range(len(docs))]))
cluster_eval(dm_mat, df_test['products'].values, 20)

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(dm_mat, df_test['products'].values, C=i))

# 3. image2vec
########################
df_test.index = df_test.id

# read all image jsons
files = '../../desktop/raw/*.json'
data = []
for file_name in glob(files):
    with open(file_name) as f:
        temp = json.load(f)
        for sku in temp:
            if sku['id'] not in df_test.id:
                continue
            obs = [df_test.loc[sku['id'], 'products']]
            obs = obs + sku['prelogits']
            data.append(obs)

# convert to ndarray
image_mat = np.array(data, dtype=object)

cluster_eval(image_mat[:, 1:], image_mat[:, 0], 20)
result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(image_mat[:, 1:], image_mat[:, 0], C=i))