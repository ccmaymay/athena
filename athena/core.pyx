from _serialization cimport FileSerializer as _FileSerializer
from _core cimport (
    LanguageModel as _LanguageModel,
    NaiveLanguageModel as _NaiveLanguageModel,
    SpaceSavingLanguageModel as _SpaceSavingLanguageModel,
    WordContextFactorization as _WordContextFactorization,
    SGD as _SGD,
    SamplingStrategy as _SamplingStrategy,
    UniformSamplingStrategy as _UniformSamplingStrategy,
    ReservoirSamplingStrategy as _ReservoirSamplingStrategy,
    EmpiricalSamplingStrategy as _EmpiricalSamplingStrategy,
    ContextStrategy as _ContextStrategy,
    StaticContextStrategy as _StaticContextStrategy,
    DynamicContextStrategy as _DynamicContextStrategy,
    LanguageModelExampleStore as _LanguageModelExampleStore,
)
from _math cimport (
    CountNormalizer as _CountNormalizer,
    ReservoirSampler as _ReservoirSampler,
)
from _sgns cimport (
    SubsamplingSGNSSentenceLearner as _SubsamplingSGNSSentenceLearner,
    SGNSSentenceLearner as _SGNSSentenceLearner,
    SGNSTokenLearner as _SGNSTokenLearner,
    SGNSModel as _SGNSModel,
)
from _word2vec cimport (
    Word2VecModel as _Word2VecModel,
)
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.utility cimport pair
from libcpp cimport bool
from libc.stddef cimport size_t
from libc.string cimport memcpy
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.operator import dereference as d
import numpy as np
cimport numpy as np

import codecs
import logging
import os
from tempfile import mkstemp


NEG_SAMPLING_STRATEGY_CHOICES = ('uniform', 'empirical', 'reservoir')
CONTEXT_STRATEGY_CHOICES = ('static', 'dynamic')
LANGUAGE_MODEL_CHOICES = ('naive', 'space-saving')


cdef vector[string] list_to_string_vector(list tokens):
    cdef vector[string] enc_tokens
    cdef string enc_token
    cdef size_t i, num_tokens

    num_tokens = len(tokens)
    enc_tokens.reserve(num_tokens)

    for i in xrange(num_tokens):
        enc_token = tokens[i].encode('utf-8')
        enc_tokens.push_back(enc_token)

    return enc_tokens


cdef list string_vector_to_list(vector[string] enc_tokens):
    cdef size_t i, num_tokens
    cdef string enc_token

    num_tokens = enc_tokens.size()
    tokens = list()

    for i in xrange(num_tokens):
        enc_token = enc_tokens[i]
        tokens.append(enc_token.decode('utf-8'))

    return tokens


cdef class SGNSModel(object):
    DEFAULT_VOCAB_DIM = int(1e5)
    DEFAULT_EMBEDDING_DIM = 100
    DEFAULT_SYMM_CONTEXT = 5
    DEFAULT_NEG_SAMPLES = 5
    DEFAULT_SUBSAMPLE_THRESHOLD = 1e-3
    DEFAULT_TAU = 1.7e7
    DEFAULT_KAPPA = 2.5e-2
    DEFAULT_RHO_LOWER_BOUND = 2.5e-6
    DEFAULT_NEG_SAMPLING_STRATEGY = 'reservoir'
    DEFAULT_CONTEXT_STRATEGY = 'dynamic'
    DEFAULT_LANGUAGE_MODEL = 'space-saving'
    DEFAULT_SMOOTHING_EXPONENT = 0.75
    DEFAULT_SMOOTHING_OFFSET = 0
    DEFAULT_REFRESH_INTERVAL = int(1e5)
    DEFAULT_REFRESH_BURN_IN = int(1e4)
    DEFAULT_RESERVOIR_SIZE = int(1e8)

    cdef shared_ptr[_SGNSModel] _model

    def get_vocab_used(self):
        return d(d(self._model).language_model).size()

    def get_vocab_dim(self):
        return d(d(self._model).factorization).get_vocab_dim()

    def get_embedding_dim(self):
        return d(d(self._model).factorization).get_embedding_dim()

    def sentence_increment(self, list tokens):
        cdef vector[string] enc_tokens
        cdef size_t i, num_tokens

        enc_tokens = list_to_string_vector(tokens)
        num_tokens = enc_tokens.size()

        for i in xrange(num_tokens):
            d(d(self._model).language_model).increment(enc_tokens[i])

    def truncate_language_model(self, size_t vocab_dim):
        d(d(self._model).language_model).truncate(vocab_dim)

    def reset_neg_sampling_strategy(self, float smoothing_exponent,
                                    float smoothing_offset):
        cdef shared_ptr[_CountNormalizer] normalizer

        normalizer = make_shared[_CountNormalizer](smoothing_exponent, smoothing_offset)
        d(d(self._model).neg_sampling_strategy).reset(
            d(d(self._model).language_model),
            d(normalizer))

    def sentence_train(self, list tokens):
        cdef vector[string] enc_tokens
        enc_tokens = list_to_string_vector(tokens)
        d(d(self._model).subsampling_sentence_learner).sentence_train(enc_tokens)

    def _get_embedding(self, long word_id):
        cdef const float* enc_embedding
        cdef np.ndarray[np.float_t, ndim=1] embedding
        cdef size_t i, embedding_dim

        embedding_dim = d(d(self._model).factorization).get_embedding_dim()
        enc_embedding = d(d(self._model).factorization).get_word_embedding(word_id)
        embedding = np.zeros((embedding_dim,), dtype=np.float)
        for i in xrange(embedding_dim):
            embedding[i] = enc_embedding[i]
        return embedding

    def get_embedding(self, unicode word):
        cdef long word_id

        word_id = d(d(self._model).language_model).lookup(
            word.encode('utf-8')
        )
        if word_id < 0:
            return None
        else:
            return self._get_embedding(word_id)

    def add_doc_embeddings(self, dict doc):
        embeddings = list()
        for token in doc['tokens']:
            embedding = self.get_embedding(token)
            if embedding is None:
                logging.warning('skipping OOV word in doc %s' %
                                doc['id'])
            else:
                embeddings.append(embedding)

        doc['embeddings'] = embeddings

    def get_word_counts(self):
        return dict(
            (
                d(d(self._model).language_model).reverse_lookup(i).decode('utf-8'),
                d(d(self._model).language_model).count(i)
            )
            for i in xrange(d(d(self._model).language_model).size())
        )

    def compute_similarity(self, string word1, string word2):
        cdef long word1_id, word2_id

        word1_id = d(d(self._model).language_model).lookup(
            word1.encode('utf-8')
        )
        if word1_id < 0:
            raise ValueError(
                'word "%s" not in language model, cannot lookup' %
                word1)

        word2_id = d(d(self._model).language_model).lookup(
            word2.encode('utf-8')
        )
        if word2_id < 0:
            raise ValueError(
                'word "%s" not in language model, cannot lookup' %
                word2)

        return d(d(self._model).token_learner).compute_similarity(
            word1_id, word2_id
        )

    def dump_similarity(self, list words, str output_path):
        cdef vector[long] word_ids
        cdef float sim
        cdef long word_id
        cdef int i, j

        word_ids.reserve(len(words))

        for i in xrange(len(words)):
            word_id = d(d(self._model).language_model).lookup(words[i].encode('utf-8'))
            if word_id < 0:
                word_ids.push_back(-1)
            else:
                word_ids.push_back(word_id)

        with codecs.open(output_path, encoding='utf-8', mode='w') as f:
            f.write(u'word.1\tword.2\tcosine\n')
            for i in xrange(len(words)):
                logging.info(u'processing word %d: %s ...' % (i, words[i]))
                for j in xrange(len(words)):
                    if i < j:
                        if word_ids[i] >= 0 and word_ids[j] >= 0:
                            sim = d(d(self._model).token_learner).compute_similarity(
                                word_ids[i], word_ids[j]
                            )
                            f.write(u'%s\t%s\t%.6g\n' %
                                    (words[i], words[j], sim))

    def get_representation(self, long left_context, long right_context,
                           list words):
        cdef vector[long] word_ids
        cdef long nn_idx

        word_ids.resize(len(words))
        for i in xrange(len(words)):
            word_ids[i] = d(d(self._model).language_model).lookup(
                words[i].encode('utf-8')
            )

        nn_idx = d(d(self._model).token_learner).find_context_nearest_neighbor_idx(
            left_context, right_context, word_ids.data()
        )

        return nn_idx

    def find_nearest_neighbor(self, unicode word):
        cdef long word_idx, nn_idx

        word_idx = d(d(self._model).language_model).lookup(word.encode('utf-8'))
        if word_idx < 0:
            logging.warning((
                u'word "%s" not in language model, '
                u'cannot lookup nearest neighbor'
            ) % word)
            return None
        nn_idx = d(d(self._model).token_learner).find_nearest_neighbor_idx(
            word_idx
        )
        if nn_idx < 0:
            logging.warning(
                u'could not find near neighbor for word "%s"' % word
            )
            return None
        return d(d(self._model).language_model).reverse_lookup(
            nn_idx
        ).decode('utf-8')

    def dump(self, str output_path):
        cdef shared_ptr[_FileSerializer[_SGNSModel]] serializer
        cdef string enc_temp_output_path

        # serialize to temporary file
        output_parent = os.path.dirname(output_path)
        (fd, temp_output_path) = mkstemp(
            dir=(output_parent if output_parent else None)
        )
        os.close(fd)
        enc_temp_output_path = temp_output_path.encode('utf-8')
        serializer = make_shared[_FileSerializer[_SGNSModel]](
            enc_temp_output_path
        )
        d(serializer).dump(d(self._model))

        # atomically re-link serialized model to permanent location
        os.rename(temp_output_path, output_path)

    @classmethod
    def load(cls, str input_path):
        cdef shared_ptr[_FileSerializer[_SGNSModel]] serializer
        cdef string enc_input_path
        enc_input_path = input_path.encode('utf-8')
        serializer = make_shared[_FileSerializer[_SGNSModel]](
            enc_input_path
        )
        model = SGNSModel()
        model._model = d(serializer).load()
        return model

    @classmethod
    def create(cls,
               string language_model=DEFAULT_LANGUAGE_MODEL,
               size_t vocab_dim=DEFAULT_VOCAB_DIM,
               size_t embedding_dim=DEFAULT_EMBEDDING_DIM,
               string context_strategy=DEFAULT_CONTEXT_STRATEGY,
               size_t symm_context=DEFAULT_SYMM_CONTEXT,
               size_t neg_samples=DEFAULT_NEG_SAMPLES,
               float subsample_threshold=DEFAULT_SUBSAMPLE_THRESHOLD,
               float tau=DEFAULT_TAU,
               float kappa=DEFAULT_KAPPA,
               float rho_lower_bound=DEFAULT_RHO_LOWER_BOUND,
               string neg_sampling_strategy=DEFAULT_NEG_SAMPLING_STRATEGY,
               float smoothing_exponent=DEFAULT_SMOOTHING_EXPONENT,
               float smoothing_offset=DEFAULT_SMOOTHING_OFFSET,
               size_t refresh_interval=DEFAULT_REFRESH_INTERVAL,
               size_t refresh_burn_in=DEFAULT_REFRESH_BURN_IN,
               size_t reservoir_size=DEFAULT_RESERVOIR_SIZE,
               bool propagate_discarded=False,
               bool propagate_retained=False,
               ):

        cdef shared_ptr[_LanguageModel] language_model_obj
        if language_model == 'naive':
            language_model_obj = <shared_ptr[_LanguageModel]> make_shared[_NaiveLanguageModel](
                subsample_threshold
            )
        elif language_model == 'space-saving':
            language_model_obj = <shared_ptr[_LanguageModel]> make_shared[_SpaceSavingLanguageModel](
                vocab_dim, subsample_threshold
            )
        else:
            raise ValueError(
                'unrecognized language model %s' % language_model
            )

        cdef shared_ptr[_SamplingStrategy] neg_sampling_strategy_obj
        if neg_sampling_strategy == 'uniform':
            neg_sampling_strategy_obj = <shared_ptr[_SamplingStrategy]> make_shared[_UniformSamplingStrategy]()
        elif neg_sampling_strategy == 'empirical':
            neg_sampling_strategy_obj = <shared_ptr[_SamplingStrategy]> make_shared[_EmpiricalSamplingStrategy](
                make_shared[_CountNormalizer](
                    smoothing_exponent, smoothing_offset
                ),
                refresh_interval, refresh_burn_in
            )
        elif neg_sampling_strategy == 'reservoir':
            neg_sampling_strategy_obj = <shared_ptr[_SamplingStrategy]> make_shared[_ReservoirSamplingStrategy](
                make_shared[_ReservoirSampler[long] ](
                    reservoir_size
                )
            )
        else:
            raise ValueError(
                'unrecognized negative sampling strategy %s' %
                neg_sampling_strategy
            )

        cdef shared_ptr[_ContextStrategy] context_strategy_obj
        if context_strategy == 'static':
            context_strategy_obj = <shared_ptr[_ContextStrategy]> make_shared[_StaticContextStrategy](
                symm_context
            )
        elif context_strategy == 'dynamic':
            context_strategy_obj = <shared_ptr[_ContextStrategy]> make_shared[_DynamicContextStrategy](
                symm_context
            )
        else:
            raise ValueError(
                'unrecognized context strategy %s' % context_strategy
            )

        model = SGNSModel()
        model._model = make_shared[_SGNSModel](
            make_shared[_WordContextFactorization](
                vocab_dim, embedding_dim
            ),
            neg_sampling_strategy_obj,
            language_model_obj,
            make_shared[_SGD](
                tau, kappa, rho_lower_bound
            ),
            context_strategy_obj,
            make_shared[_SGNSTokenLearner](),
            make_shared[_SGNSSentenceLearner](
                neg_samples, propagate_retained
            ),
            make_shared[_SubsamplingSGNSSentenceLearner](
                propagate_discarded
            )
        )
        d(d(model._model).token_learner).set_model(model._model)
        d(d(model._model).sentence_learner).set_model(model._model)
        d(d(model._model).subsampling_sentence_learner).set_model(model._model)
        return model

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--language-model', type=str,
                            choices=LANGUAGE_MODEL_CHOICES,
                            help='language model type',
                            default=cls.DEFAULT_LANGUAGE_MODEL)
        parser.add_argument('--symm-context', type=int,
                            help='max number of context words to use '
                                 '(on each side, so --symm-context 2 '
                                 'means 2 words on left, 2 on right)',
                            default=cls.DEFAULT_SYMM_CONTEXT)
        parser.add_argument('--neg-samples', type=int,
                            help='number of negative samples to draw',
                            default=cls.DEFAULT_NEG_SAMPLES)
        parser.add_argument('--tau', type=float,
                            help='base offset of learning rate',
                            default=cls.DEFAULT_TAU)
        parser.add_argument('--kappa', type=float,
                            help='exponent of learning rate',
                            default=cls.DEFAULT_KAPPA)
        parser.add_argument('--rho-lower-bound', type=float,
                            help='lower bound of learning rate',
                            default=cls.DEFAULT_RHO_LOWER_BOUND)
        parser.add_argument('--vocab-dim', type=int,
                            help='dimension of vocabulary',
                            default=cls.DEFAULT_VOCAB_DIM)
        parser.add_argument('--embedding-dim', type=int,
                            help='dimension of embedding',
                            default=cls.DEFAULT_EMBEDDING_DIM)
        parser.add_argument('--subsample-threshold', type=float,
                            help='frequent word subsampling threshold',
                            default=cls.DEFAULT_SUBSAMPLE_THRESHOLD)
        parser.add_argument('--neg-sampling-strategy', type=str,
                            choices=NEG_SAMPLING_STRATEGY_CHOICES,
                            help='negative sampling strategy',
                            default=cls.DEFAULT_NEG_SAMPLING_STRATEGY)
        parser.add_argument('--context-strategy', type=str,
                            choices=CONTEXT_STRATEGY_CHOICES,
                            help='context size strategy',
                            default=cls.DEFAULT_CONTEXT_STRATEGY)
        parser.add_argument('--smoothing-exponent', type=float,
                            help='language count smoothing exponent',
                            default=cls.DEFAULT_SMOOTHING_EXPONENT)
        parser.add_argument('--smoothing-offset', type=float,
                            help='language count smoothing offset',
                            default=cls.DEFAULT_SMOOTHING_OFFSET)
        parser.add_argument('--refresh-interval', type=int,
                            help='language model refresh interval',
                            default=cls.DEFAULT_REFRESH_INTERVAL)
        parser.add_argument('--refresh-burn-in', type=int,
                            help='language model refresh burn-in',
                            default=cls.DEFAULT_REFRESH_BURN_IN)
        parser.add_argument('--reservoir-size', type=int,
                            help='reservoir (negative) sampler size',
                            default=cls.DEFAULT_RESERVOIR_SIZE)
        parser.add_argument('--propagate-discarded',
                            action='store_true',
                            help='propagate words discarded from '
                                 'subsampling to language model')
        parser.add_argument('--propagate-retained',
                            action='store_true',
                            help='propagate words retained in '
                                 'subsampling to language model')

    @classmethod
    def from_namespace(cls, ns):
        return cls.create(
            language_model=ns.language_model,
            vocab_dim=ns.vocab_dim,
            embedding_dim=ns.embedding_dim,
            context_strategy=ns.context_strategy,
            symm_context=ns.symm_context,
            neg_samples=ns.neg_samples,
            subsample_threshold=ns.subsample_threshold,
            tau=ns.tau,
            kappa=ns.kappa,
            rho_lower_bound=ns.rho_lower_bound,
            neg_sampling_strategy=ns.neg_sampling_strategy,
            smoothing_exponent=ns.smoothing_exponent,
            smoothing_offset=ns.smoothing_offset,
            refresh_interval=ns.refresh_interval,
            refresh_burn_in=ns.refresh_burn_in,
            reservoir_size=ns.reservoir_size,
            propagate_discarded=ns.propagate_discarded,
            propagate_retained=ns.propagate_retained,
        )


ctypedef pair[string, pair[vector[string], vector[string]]] _MultiLabelDoc


cdef _MultiLabelDoc dict_to_multi_label_doc(dict doc):
    cdef _MultiLabelDoc enc_doc
    cdef string enc_doc_id
    cdef vector[string] enc_tokens
    cdef vector[string] enc_labels

    enc_doc_id = doc['id'].encode('utf-8')
    enc_tokens = list_to_string_vector(list(doc['tokens']))
    enc_labels = list_to_string_vector(list(doc['labels']))

    enc_doc.first = enc_doc_id
    enc_doc.second.first = enc_tokens
    enc_doc.second.second = enc_labels

    return enc_doc


cdef dict multi_label_doc_to_dict(_MultiLabelDoc enc_doc):
    cdef string enc_doc_id
    cdef vector[string] enc_tokens
    cdef vector[string] enc_labels

    enc_doc_id = enc_doc.first
    enc_tokens = enc_doc.second.first
    enc_labels = enc_doc.second.second

    return dict(
        id=enc_doc_id.decode('utf-8'),
        tokens=string_vector_to_list(enc_tokens),
        labels=string_vector_to_list(enc_labels),
    )


cdef class LMMultiLabelDocStore(object):
    DEFAULT_VOCAB_DIM = int(1e2)
    DEFAULT_SUBSAMPLE_THRESHOLD = 1e-3
    DEFAULT_LANGUAGE_MODEL = 'space-saving'
    DEFAULT_NUM_EXAMPLES_PER_WORD = int(1e5)

    cdef shared_ptr[_LanguageModelExampleStore[_MultiLabelDoc]] _example_store

    def get_vocab_used(self):
        return d(self._example_store).get_language_model().size()

    def increment(self, dict doc):
        cdef string enc_label
        cdef _MultiLabelDoc enc_doc
        cdef int i, num_labels

        enc_doc = dict_to_multi_label_doc(doc)
        num_labels = len(doc['labels'])

        for i in xrange(num_labels):
            enc_label = doc['labels'][i].encode('utf-8')
            d(self._example_store).increment(enc_label, enc_doc)

    def get_word_counts(self):
        return dict(
            (
                d(self._example_store).get_language_model().reverse_lookup(i).decode('utf-8'),
                d(self._example_store).get_language_model().count(i)
            )
            for i in xrange(d(self._example_store).get_language_model().size())
        )

    def get_docs(self, unicode label):
        cdef string enc_label
        cdef long label_idx

        enc_label = label.encode('utf-8')
        label_idx = d(self._example_store).get_language_model().lookup(enc_label)

        if label_idx < 0:
            raise ValueError(
                u'label "%s" not in language model, cannot lookup' %
                label)

        return self._get_docs(label_idx)

    def _get_docs(self, long label_idx):
        cdef const _ReservoirSampler[_MultiLabelDoc]* enc_examples
        cdef size_t i

        enc_examples = &(d(self._example_store).get_examples(label_idx))

        docs = list()
        for i in xrange(d(enc_examples).filled_size()):
            docs.append(multi_label_doc_to_dict(d(enc_examples)[i]))

        return docs

    def dump(self, str output_path):
        cdef shared_ptr[_FileSerializer[_LanguageModelExampleStore[_MultiLabelDoc]]] serializer
        cdef string enc_temp_output_path

        # serialize to temporary file
        output_parent = os.path.dirname(output_path)
        (fd, temp_output_path) = mkstemp(
            dir=(output_parent if output_parent else None)
        )
        os.close(fd)
        enc_temp_output_path = temp_output_path.encode('utf-8')
        serializer = make_shared[_FileSerializer[_LanguageModelExampleStore[_MultiLabelDoc]]](
            enc_temp_output_path
        )
        d(serializer).dump(d(self._example_store))

        # atomically re-link serialized model to permanent location
        os.rename(temp_output_path, output_path)

    @classmethod
    def load(cls, str input_path):
        cdef shared_ptr[_FileSerializer[_LanguageModelExampleStore[_MultiLabelDoc]]] serializer
        cdef string enc_input_path
        enc_input_path = input_path.encode('utf-8')
        serializer = make_shared[_FileSerializer[_LanguageModelExampleStore[_MultiLabelDoc]]](
            enc_input_path
        )
        example_store = LMMultiLabelDocStore()
        example_store._example_store = d(serializer).load()
        return example_store

    @classmethod
    def create(cls,
               string language_model=DEFAULT_LANGUAGE_MODEL,
               size_t vocab_dim=DEFAULT_VOCAB_DIM,
               float subsample_threshold=DEFAULT_SUBSAMPLE_THRESHOLD,
               size_t num_examples_per_word=DEFAULT_NUM_EXAMPLES_PER_WORD,
               ):

        cdef shared_ptr[_LanguageModel] language_model_obj
        if language_model == 'naive':
            language_model_obj = <shared_ptr[_LanguageModel]> make_shared[_NaiveLanguageModel](
                subsample_threshold
            )
        elif language_model == 'space-saving':
            language_model_obj = <shared_ptr[_LanguageModel]> make_shared[_SpaceSavingLanguageModel](
                vocab_dim, subsample_threshold
            )
        else:
            raise ValueError(
                'unrecognized language model %s' % language_model
            )

        example_store = LMMultiLabelDocStore()
        example_store._example_store = make_shared[_LanguageModelExampleStore[_MultiLabelDoc]](
            language_model_obj,
            num_examples_per_word
        )
        return example_store

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--doc-store-language-model', type=str,
                            choices=LANGUAGE_MODEL_CHOICES,
                            help='language model example store: '
                                 'language model type',
                            default=cls.DEFAULT_LANGUAGE_MODEL)
        parser.add_argument('--doc-store-vocab-dim', type=int,
                            help='language model example store: '
                                 'dimension of vocabulary',
                            default=cls.DEFAULT_VOCAB_DIM)
        parser.add_argument('--doc-store-subsample-threshold', type=float,
                            help='language model example store: '
                                 'frequent word subsampling threshold',
                            default=cls.DEFAULT_SUBSAMPLE_THRESHOLD)
        parser.add_argument('--doc-store-num-examples-per-word', type=int,
                            help='language model example store: '
                                 'no. examples to store per word',
                            default=cls.DEFAULT_NUM_EXAMPLES_PER_WORD)

    @classmethod
    def from_namespace(cls, ns):
        return cls.create(
            language_model=ns.doc_store_language_model,
            vocab_dim=ns.doc_store_vocab_dim,
            subsample_threshold=ns.doc_store_subsample_threshold,
            num_examples_per_word=ns.doc_store_num_examples_per_word,
        )


cdef class Word2VecModel(object):
    cdef shared_ptr[_Word2VecModel] _model

    def get_vocab_dim(self):
        return d(self._model).vocab_dim

    def get_embedding_dim(self):
        return d(self._model).embedding_dim

    def get_word_embeddings(self):
        cdef long long vocab_dim, embedding_dim
        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] m

        vocab_dim = d(self._model).vocab_dim
        embedding_dim = d(self._model).embedding_dim
        m = np.ascontiguousarray(np.zeros((vocab_dim, embedding_dim),
                                          dtype=np.float32),
                                 dtype=np.float32)

        memcpy(<float*> m.data,
               d(self._model).word_embeddings.data(),
               vocab_dim * embedding_dim * sizeof(float))

        return m

    def get_vocab(self):
        return list(
            d(self._model).vocab[i]
            for i in xrange(d(self._model).vocab_dim)
        )

    @classmethod
    def load(cls, str input_path):
        cdef shared_ptr[_FileSerializer[_Word2VecModel]] serializer
        cdef string enc_input_path
        enc_input_path = input_path.encode('utf-8')
        serializer = make_shared[_FileSerializer[_Word2VecModel]](
            enc_input_path
        )
        model = Word2VecModel()
        model._model = d(serializer).load()
        return model
