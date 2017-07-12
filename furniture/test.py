import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale
from gensim.models.keyedvectors import KeyedVectors
from preprocess import data_load, pre_process
from doc2vec import text_process, doc2vec_centroid
from doc2vec_weight import doc_to_vec
from tf_idf import title_process
from evaluation import cluster_eval, class_eval



# load raw_data
raw_data = data_load()

# pre_process data
df, fts = pre_process(raw_data)

# use group size > 55 for testing
df_test = df.dropna(subset=['products'])
df_test = df_test.groupby('products').filter(lambda x: len(x) > 55)

# return titles array
titles = df_test.title.values


# 1. tfidf test
################
tfidf_mat = title_process(titles)
tfidf_df = pd.DataFrame(tfidf_mat)
tfidf_df['products'] = df_test['products'].values
# tfidf_df.to_csv("lsi_mat.csv", sep=",", index=False)
cluster_eval(tfidf_df, 70)

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100]
for i in seq:
    result.append(class_eval(tfidf_df, C=i))

# 2. doc2vec test
################
# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format('../../Desktop/trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec')

# return processed titles bag of words
docs = [text_process(title) for title in titles]
docs = np.array(docs)


# 2.1 centroid method
####################
centroid_mat = normalize(np.array([doc2vec_centroid(doc, model_ft.wv) for doc in docs]))
centroid_df = pd.DataFrame(centroid_mat)
centroid_df['products'] = df_test['products'].values
# centroid_df.to_csv("centroid_mat.csv", sep=",", index=False)
cluster_eval(centroid_df, 70)

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100]
for i in seq:
    result.append(class_eval(centroid_df, C=i))


# 2.2 weighted average method
######################
weight_mat = doc_to_vec(docs=docs, model=model_ft, algo='tfidf', pca=1)
weight_mat = normalize(weight_mat)
weight_df = pd.DataFrame(weight_mat)
weight_df['products'] = df_test['products'].values
# weight_df.to_csv("wa_mat.csv", sep=",", index=False)
cluster_eval(weight_df, 70)

# result = []
# seq = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
# for i in seq:
#     weight_df = pd.DataFrame(doc_to_vec(docs=docs, model=model_ft, fun=get_word_frequency, a=i))
#     weight_df['products'] = df_test['products'].values
#     result.append(cluster_eval(weight_df, 70))

result = []
seq = [0.1, 1, 2, 5, 10, 20, 50, 100, 1000]
for i in seq:
    result.append(class_eval(weight_df, C=i))


# 3. Gensim doc2vec
########################

