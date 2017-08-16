from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument, Doc2Vec
from gensim.models.word2vec import LineSentence, Word2Vec
import fasttext
import time


from preprocess import data_load, pre_process, text_process


input = '../dat/data/18000*.json'
filename = '../dat/cleaned_text_plus_wp.txt'
doc2vec_model = '../trained_models/dm_win4.model'
word2vec_model = '../trained_models/word2vec_win1.model'
fasttext_model = '../trained_models/ft_win3'
trained_model_full = '../trained_models/dm.model_full'

# train original word2vec
sentences = LineSentence(filename)
model = Word2Vec(sentences, size=300, window=3, min_count=5, workers=6, sg=1)
model.save(word2vec_model)


# train fasttext
model = fasttext.skipgram(filename, fasttext_model , dim=300, ws=5, word_ngrams=3)


# train doc2vec
# build TaggedLineDocument object
documents = TaggedLineDocument(filename)
dm_model = Doc2Vec(size=300, window=4, min_count=5, workers=6)

# build vocab for model
dm_model.build_vocab(documents)

# train model using corpus
dm_model.train(documents, total_examples=dm_model.corpus_count, epochs=10)

# save model
dm_model.save(doc2vec_model)



# # learn from corpus + titles data
# # build list of TaggedDocument objects from corpus
# alldocs = []
# with open(filename) as alldata:
#     for line_no, line in enumerate(alldata):
#         words = line.split()
#         alldocs.append(TaggedDocument(words, ['c_%s' % line_no]))
#
# # load raw_data
# raw_data = data_load(input)
#
# # pre_process data
# df, fts = pre_process(raw_data)
#
# # build list of TaggedDocument objects from titles
# titles = df.title.values
# docs = [text_process(title, trained_model_full) for title in titles]
#
# for i, row in enumerate(docs):
#     alldocs.append(TaggedDocument(row, ['t_%s' % i]))
#
# # set model parameters
# dm_model_full = Doc2Vec(size=300, window=5, min_count=5, workers=6)
#
# # build vocab for model
# dm_model_full.build_vocab(alldocs)
#
# # train model using corpus
# start = time.time()
# dm_model_full.train(alldocs, total_examples=dm_model_full.corpus_count, epochs=10)
# print(time.time()-start)
#
# # save model
# dm_model_full.save(trained_model_full)