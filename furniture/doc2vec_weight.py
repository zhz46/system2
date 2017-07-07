import numpy as np
from sklearn.decomposition import PCA


# count = 0
# for i in model_ft.vocab:
#     count += model_ft.vocab[i].count

def get_word_frequency(word, model):
    return model.vocab[word].count/51576208551


# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# convert a list of sentence with word2vec items into a set of sentence vectors
def doc_to_vec(docs, model, embedding_size=300, a = 1e-3):

    doc_set = []
    for doc in docs:
        doc = [word for word in doc if word in model.vocab]
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        doc_length = len(doc)
        word_vectors = model.wv[doc]
        for i in np.arange(doc_length):
            a_value = a / (a + get_word_frequency(doc[i], model))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_vectors[i]))  # vs += sif * word_vector

        vs = np.divide(vs, doc_length)  # weighted average
        doc_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this doc set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(doc_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    doc_vecs = []
    for vs in doc_set:
        sub = np.multiply(u,vs)
        doc_vecs.append(np.subtract(vs, sub))

    return np.array(doc_vecs)
