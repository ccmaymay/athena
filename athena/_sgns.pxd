from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from _core cimport (
    LanguageModel, WordContextFactorization, SGD,
    SamplingStrategy, ContextStrategy
)


cdef extern from "_sgns.h":
    cdef cppclass SGNSModel:
        shared_ptr[WordContextFactorization] factorization
        shared_ptr[SamplingStrategy] neg_sampling_strategy
        shared_ptr[LanguageModel] language_model
        shared_ptr[SGD] sgd
        shared_ptr[ContextStrategy] ctx_strategy
        shared_ptr[SGNSTokenLearner] token_learner
        shared_ptr[SGNSSentenceLearner] sentence_learner
        shared_ptr[SubsamplingSGNSSentenceLearner] subsampling_sentence_learner

        SGNSModel(
            shared_ptr[WordContextFactorization] factorization_,
            shared_ptr[SamplingStrategy] neg_sampling_strategy_,
            shared_ptr[LanguageModel] language_model_,
            shared_ptr[SGD] sgd_,
            shared_ptr[ContextStrategy] ctx_strategy_,
            shared_ptr[SGNSTokenLearner] token_learner_,
            shared_ptr[SGNSSentenceLearner] sentence_learner_,
            shared_ptr[SubsamplingSGNSSentenceLearner]
              subsampling_sentence_learner_)

    cdef cppclass SGNSTokenLearner:
        SGNSTokenLearner()
        float compute_similarity(size_t word1_idx, size_t word2_idx)
        long find_nearest_neighbor_idx(size_t word_idx)
        long find_context_nearest_neighbor_idx(size_t left_context,
                                               size_t right_context,
                                               const long *word_ids)
        void set_model(shared_ptr[SGNSModel] model)

    cdef cppclass SGNSSentenceLearner:
        SGNSSentenceLearner(size_t neg_samples,
                            bool propagate_retained)
        void sentence_train(const vector[string]& words) except +
        void set_model(shared_ptr[SGNSModel] model)

    cdef cppclass SubsamplingSGNSSentenceLearner:
        SubsamplingSGNSSentenceLearner(bool propagate_discarded)
        void sentence_train(const vector[string]& words) except +
        void set_model(shared_ptr[SGNSModel] model)
