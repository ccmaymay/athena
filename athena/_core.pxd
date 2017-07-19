from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.utility cimport pair
from _math cimport CountNormalizer, ReservoirSampler


cdef extern from "_core.h":
    cdef cppclass LanguageModel:
        pair[long,string] increment(const string& word)
        long lookup(const string& word)
        string reverse_lookup(long word_idx) except +
        size_t count(long word_idx)
        void truncate(size_t max_size) except +
        size_t size()

    cdef cppclass NaiveLanguageModel:
        NaiveLanguageModel(float subsample_threshold)
        pair[long,string] increment(const string& word)
        long lookup(const string& word)
        string reverse_lookup(long word_idx) except +
        size_t count(long word_idx)
        size_t size()
        void truncate(size_t max_size) except +

    cdef cppclass SpaceSavingLanguageModel:
        SpaceSavingLanguageModel(size_t num_counters,
                                 float subsample_threshold)
        pair[long,string] increment(const string& word)
        long lookup(const string& word)
        string reverse_lookup(long word_idx) except +
        size_t count(long word_idx)
        size_t size()
        void truncate(size_t max_size) except +
        size_t capacity()

    cdef cppclass ContextStrategy:
        pass

    cdef cppclass StaticContextStrategy:
        StaticContextStrategy(size_t symm_context)

    cdef cppclass DynamicContextStrategy:
        DynamicContextStrategy(size_t symm_context)

    cdef cppclass SamplingStrategy:
        void reset(const LanguageModel& language_model,
                   const CountNormalizer& normalizer)

    cdef cppclass UniformSamplingStrategy:
        UniformSamplingStrategy()
        void reset(const LanguageModel& language_model,
                   const CountNormalizer& normalizer)

    cdef cppclass EmpiricalSamplingStrategy:
        EmpiricalSamplingStrategy(shared_ptr[CountNormalizer] normalizer,
                                  size_t refresh_interval,
                                  size_t refresh_burn_in)
        void reset(const LanguageModel& language_model,
                   const CountNormalizer& normalizer)

    cdef cppclass ReservoirSamplingStrategy:
        ReservoirSamplingStrategy(shared_ptr[ReservoirSampler[long] ] sampler)
        void reset(const LanguageModel& language_model,
                   const CountNormalizer& normalizer)

    cdef cppclass WordContextFactorization:
        WordContextFactorization(size_t vocab_dim,
                                 size_t embedding_dim)
        size_t get_embedding_dim()
        size_t get_vocab_dim()
        float* get_word_embedding(size_t word_idx)
        float* get_context_embedding(size_t word_idx)

    cdef cppclass SGD:
        SGD(float tau, float kappa, float rho_lower_bound)

    cdef cppclass LanguageModelExampleStore[T]:
        LanguageModelExampleStore(shared_ptr[LanguageModel] language_model,
                                  size_t num_examples_per_word)
        pair[long,string] increment(const string& word,
                                    const T& example)
        const LanguageModel& get_language_model()
        const ReservoirSampler[T]& get_examples(long idx)
