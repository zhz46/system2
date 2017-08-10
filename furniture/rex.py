import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import Doc2Vec

from preprocess import data_load, pre_process, image_merge, df_filter, title_process, text_process, doc2vec_tfidf, doc2vec_centroid
from rex_tools import query, parallel

text_input = '../dat/data/18000*.json'
image_input = '../dat/raw/18000*.json'
# text_input = '/srv/zz/temp/18000*.json'
# image_input = '/yg/analytics/rex/tensorflow/image2vec/dat/output/raw/18000*.json'
rex_output = '../output/old_sim.json'
data_output = '../output/data.json'
word2vec_model = '../trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_3_wordNgrams_3_ws_5.vec'
doc2vec_model = '../trained_models/dm.model'


model_pars = {'weight': {'title_wt': 0.2,
                         'prod_wt': 0.5,
                         'image_wt': 0.0,
                         'brand_wt':0.2,
                         'price_wt':0.1},
              'method': 'lsa',
              'title_dim': 300}


# load raw_data
raw_data = data_load(text_input)

# pre_process data
df = pre_process(raw_data)

# load model
if model_pars['method'] in ['mean_word2vec', 'tfidf_word2vec']:
    model_path = word2vec_model
    model_ft = KeyedVectors.load_word2vec_format(model_path)
    # filter out empty bags of word
    df = df_filter(df, model_ft)
if model_pars['method'] == 'dm':
    model_path = doc2vec_model
    model_ft = Doc2Vec.load(model_path)

# inner join text and image data
if model_pars['weight']['image_wt'] != 0:
    df = image_merge(df, image_input)

# sort df by category
df = df.sort_values(by='category_id').reset_index(drop=True)

# build map
category_map = df.groupby('category_id').groups
for key, value in category_map.items():
    value = value[0], value[-1] + 1
    category_map[key] = value

# return titles array
titles = df.title.values

if model_pars['method'] == 'lsa':
    title_mat = title_process(titles)
elif model_pars['method'] == 'tfidf_word2vec':
    docs = np.array([text_process(title, model_ft) for title in titles])
    title_mat = normalize(doc2vec_tfidf(docs, model_ft))
    # title_mat = normalize(doc_to_vec(docs=docs, model=model_ft, algo='tfidf', pca=1))
elif model_pars['method'] == 'mean_word2vec':
    docs = np.array([text_process(title, model_ft) for title in titles])
    title_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
elif model_pars['method'] == 'dm':
    docs = [text_process(title, model_ft) for title in titles]
    title_mat = normalize(np.array([model_ft.infer_vector(doc) for doc in docs]))
else:
    raise Exception("ERROR: " + model_pars['method'] + ' not found')

# build feature index
features = ['id', 'group_id', 'category_id']
if model_pars['weight']['prod_wt'] != 0:
    features.extend(['products', 'parentProducts'])
if model_pars['weight']['brand_wt'] != 0:
    features.append('brand')
if model_pars['weight']['price_wt'] != 0:
    features.append('price_hint')
fts = {feature: id for id, feature in enumerate(features)}
fts['title'] = (len(fts), len(fts) + model_pars['title_dim'])

# need to make a copy before concatenate!!, 20 times faster
index_mat = df[features].values.copy()

# expand prelogits into its own dataframe
if model_pars['weight']['image_wt'] != 0:
    df_prelogit = df.prelogits.apply(pd.Series)
    prelogit_mat = normalize(df_prelogit.values)
    mat = np.concatenate((index_mat, title_mat, prelogit_mat), axis=1)
    fts['image'] = fts['title'][1]
else:
    mat = np.concatenate((index_mat, title_mat), axis=1)

# make a wrapper of query function
def query_wrapper(ind):
    return query(ind, k=48, data=mat, fts=fts, map=category_map, method_name=model_pars['method'], wt=model_pars['weight'])

# generate overall recommendation pair list
start = time.time()
rs_output = parallel(query_wrapper, range(len(df)), 6)
print(time.time() - start)

# output content_rs.json
with open(rex_output, 'w') as f:
    json.dump(rs_output, f)


fts_output = ['id', 'brand', 'image_url', 'title', 'category_id']
df_output = df[fts_output]
df_output = df_output.rename(columns={'category_id':'node_id'})
df_output.to_json(data_output, orient='records')