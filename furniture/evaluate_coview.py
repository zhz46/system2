import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import Doc2Vec

from preprocess import data_load, pre_process, title_process, image_merge, df_filter, text_process, doc2vec_centroid, product_map, doc2vec_tfidf
from evaluate_tools import generate_map, df_sort, score, parallel


co_view_query = '../dat/co_view_query.pkl'
# text_input = '../dat/data/18000*.json'
# image_input = '../dat/raw/18000*.json'
text_input = '/srv/zz/temp/18000*.json'
image_input = '/yg/analytics/rex/tensorflow/image2vec/dat/output/raw/18000*.json'
word2vec_model = '../trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_3_wordNgrams_3_ws_5.vec'
doc2vec_model = '../trained_models/dm.model'


# co_view = co_view_select(sqlite_file)
# with open('../dat/co_view_query.pkl', 'wb') as output:
#     pickle.dump(co_view, output, pickle.HIGHEST_PROTOCOL)

# text + image score
model_pars = {'weight': {'title_wt': 0.2,
                         'prod_wt': 0.5,
                         'image_wt': 0.0,
                         'brand_wt':0.2,
                         'price_wt':0.1},
              'method': 'lsa',
              'title_dim': 300}

with open(co_view_query, 'rb') as f:
    co_view = pickle.load(f)

# load raw_data
raw_data = data_load(text_input)

# pre_process data
df, fts = pre_process(raw_data)

if model_pars['method'] in ['mean_word2vec', 'tfidf_word2vec']:
    model_path = word2vec_model
    model_ft = KeyedVectors.load_word2vec_format(model_path)
    # filter out empty bags of word
    df, _ = df_filter(df, model_ft)
if model_pars['method'] == 'dm':
    model_path = doc2vec_model
    model_ft = Doc2Vec.load(model_path)

# inner join text and image data
if model_pars['images']:
    df = image_merge(df, image_input)

# generate maps
co_view_map, sec2pri = generate_map(df, co_view)

# sort df by category_id
new_df, category_map = df_sort(df)

# return titles array
titles = new_df.title.values

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
features = ['sku_id', 'group_id', 'category_id']
if model_pars['products']:
    features.extend(['products', 'parentProducts'])
fts = {feature: id for id, feature in enumerate(features)}
fts['title'] = (len(fts), len(fts) + model_pars['title_dim'])

# need to make a copy before concatenate!!, 20 times faster
index_mat = new_df[features].values.copy()

# expand prelogits into its own dataframe
if model_pars['weight']['image_wt'] != 0:
    df_prelogit = new_df.prelogits.apply(pd.Series)
    prelogit_mat = normalize(df_prelogit.values)
    mat = np.concatenate((index_mat, title_mat, prelogit_mat), axis=1)
    fts['images'] = fts['title'][1]
else:
    mat = np.concatenate((index_mat, title_mat), axis=1)

# divide mats
candidate_mat = mat[:(len(df) - len(sec2pri))].copy()
second_mat = mat[(len(df) - len(sec2pri)):].copy()

# calculate importance score for each sku
for i in [0.2,0.3]:
    start = time.time()
    def score_combo(idx):
        return score(idx, candidate_mat=candidate_mat, sec2pri=sec2pri, second_mat=second_mat, fts=fts,
              category_map=category_map, co_view_map=co_view_map, model_pars=model_pars)

    combo_score = parallel(score_combo, list(co_view_map.keys()))
    combo_score = np.array(combo_score)
    ratio = np.sum(combo_score, axis=0)[0] / np.sum(combo_score, axis=0)[1]
    print(ratio)
    print(time.time()-start)