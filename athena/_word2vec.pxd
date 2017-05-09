from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "_word2vec.h":
    cdef cppclass Word2VecModel:
        long long vocab_dim
        long long embedding_dim
        vector[string] vocab
        vector[float] word_embeddings

        Word2VecModel()
