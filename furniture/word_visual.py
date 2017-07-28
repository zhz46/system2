import re
import numpy as np
from tsne import bh_sne
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from preprocess import data_load, pre_process


# load raw_data
raw_data = data_load()

# pre_process data
df, fts = pre_process(raw_data)

# pre-trained model from fasttext
model_ft = KeyedVectors.load_word2vec_format(
    '../../Desktop/trained_models/titles_wp_model_dim_300_maxn_6_minCount_5_minn_1.vec')

raw_list = list(df.products.value_counts()[:80].index)
raw_list.extend(['playstation', 'xbox', 'iphone', 'xiaomi', 'console', 'cat', 'dog'])
word_list = []
filter_list = []
for i in range(len(raw_list)):
    tokens = re.split(' and |\s', raw_list[i])
    if all([token in model_ft.vocab for token in tokens]):
        word_list.append(tokens)
        filter_list.append(raw_list[i])

# generate word vectors
word_arr = np.zeros((len(word_list), 300))
for i in range(len(word_list)):
    word_arr[i, :] = normalize(np.mean(model_ft[word_list[i]], axis=0))

# tsne
reduce_arr = bh_sne(word_arr, d=2, perplexity=25)

# plot
vis_x = reduce_arr[:, 0]
vis_y = reduce_arr[:, 1]
fig, ax = plt.subplots(figsize=(30, 10))
ax.scatter(vis_x, vis_y)
for i, txt in enumerate(filter_list):
    ax.annotate(txt, (vis_x[i],vis_y[i]), size=8)


