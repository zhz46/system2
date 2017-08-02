from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument, Doc2Vec
import time

from preprocess import data_load, pre_process, text_process


input = '../dat/data/18000*.json'
filename = '../dat/cleaned_text_plus_wp.txt'
trained_model = '../trained_models/dm.model'
trained_model_full = '../trained_models/dm.model_full'

# learn from corpus
# build TaggedLineDocument object
documents = TaggedLineDocument(filename)

# set model parameters
dm_model = Doc2Vec(size=300, window=5, min_count=5, workers=6)

# build vocab for model
dm_model.build_vocab(documents)

# train model using corpus
start = time.time()
dm_model.train(documents, total_examples=dm_model.corpus_count, epochs=10)
print(time.time()-start)

# save model
dm_model.save(trained_model)


# learn from corpus + titles data
# build list of TaggedDocument objects from corpus
alldocs = []
with open(filename) as alldata:
    for line_no, line in enumerate(alldata):
        words = line.split()
        alldocs.append(TaggedDocument(words, ['c_%s' % line_no]))

# load raw_data
raw_data = data_load(input)

# pre_process data
df, fts = pre_process(raw_data)

# build list of TaggedDocument objects from titles
titles = df.title.values
docs = [text_process(title) for title in titles]

for i, row in enumerate(docs):
    alldocs.append(TaggedDocument(row, ['t_%s' % i]))

# set model parameters
dm_model_full = Doc2Vec(size=300, window=5, min_count=5, workers=6)

# build vocab for model
dm_model_full.build_vocab(alldocs)

# train model using corpus
start = time.time()
dm_model_full.train(alldocs, total_examples=dm_model_full.corpus_count, epochs=10)
print(time.time()-start)

# save model
dm_model_full.save(trained_model_full)