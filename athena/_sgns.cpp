#include "_core.h"
#include "_log.h"
#include "_math.h"
#include "_serialization.h"
#include "_sgns.h"

#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <cstring>
#include <utility>


using namespace std;





/*
void SGNSTokenLearner::serialize(ostream& stream) const {
  Serializer<WordContextFactorization>::serialize(*factorization, stream);
  Serializer<ReservoirSamplingStrategy>::serialize(*neg_sampling_strategy, stream);
  Serializer<NaiveLanguageModel>::serialize(*language_model, stream);
  Serializer<SGD>::serialize(*sgd, stream);
}

SGNSTokenLearner&& SGNSTokenLearner::deserialize(istream& stream) {
  auto factorization_(
      Serializer<WordContextFactorization>::deserialize(stream));
  auto neg_sampling_strategy_(
      Serializer<ReservoirSamplingStrategy>::deserialize(stream));
  auto language_model_(
      Serializer<NaiveLanguageModel>::deserialize(stream));
  auto sgd_(Serializer<SGD>::deserialize(stream));
  return std::move(SGNSTokenLearner(
    factorization_,
    neg_sampling_strategy_,
    language_model_,
    sgd_
  ));
}

bool SGNSTokenLearner::equals(const SGNSTokenLearner& other) const {
  return factorization->equals(*(other.factorization)) &&
    neg_sampling_strategy->equals(*(other.neg_sampling_strategy)) &&
    language_model->equals(*(other.language_model)) &&
    sgd->equals(*(other.sgd));
}
*/


/*
//
// SGNSSentenceLearner
//


void SGNSSentenceLearner::increment(const string& word) {
  const pair<long,string> ejectee =
    token_learner->language_model->increment(word);
  const long ejectee_idx = ejectee.first;
  if (ejectee_idx >= 0) {
    token_learner->reset_word(ejectee_idx);
  }
  token_learner->neg_sampling_strategy->step(*(token_learner->language_model),
                                      token_learner->language_model->lookup(word));
}

void SGNSSentenceLearner::sentence_train(const vector<string>& words) {
  // (optionally) add all words in sentence to language model
  if (_propagate_retained)
    for (auto it = words.begin(); it != words.end(); ++it)
      increment(*it);

  // compute in-vocabulary subsequence (dirty: effective context size
  // grows to skip over out-of-vocabulary words)
  vector<long> word_ids;
  word_ids.reserve(words.size());
  size_t num_oov = 0;
  for (auto it = words.begin(); it != words.end(); ++it) {
    long word_id = token_learner->language_model->lookup(*it);
    if (word_id >= 0) {
      word_ids.push_back(word_id);
    } else {
      ++num_oov;
    }
  }
  debug(__func__, "skipping " << num_oov << " OOV words in sentence\n");

  // loop over all contexts, training on each non-empty one
  for (size_t target_word_pos = 0; target_word_pos < word_ids.size();
       ++target_word_pos) {
    // compute context size
    const pair<size_t,size_t> ctx_size = ctx_strategy->size(
      target_word_pos, (word_ids.size() - 1) - target_word_pos
    );
    const size_t left_ctx = ctx_size.first;
    const size_t right_ctx = ctx_size.second;

    // if computed context is non-empty, train
    if (left_ctx > 0 || right_ctx > 0) {
      const size_t ctx_size = left_ctx + 1 + right_ctx;
      const size_t ctx_start = target_word_pos - left_ctx;
      const size_t ctx_end = ctx_start + ctx_size;

      // train on current context
      for (size_t i = ctx_start; i < ctx_end; ++i) {
        // for all context (output) words...
        if (i != target_word_pos) {
          token_learner->token_train(word_ids[target_word_pos],
                                            word_ids[i], _neg_samples);
        }
      }
      token_learner->sgd->step(word_ids[target_word_pos]);
    }
  }
}
*/

/*
void SGNSSentenceLearner::serialize(ostream& stream) const {
  Serializer<SGNSTokenLearner>::serialize(*token_learner, stream);
  Serializer<ContextStrategy>::serialize(*ctx_strategy, stream);
  Serializer<size_t>::serialize(_neg_samples, stream);
  Serializer<bool>::serialize(_propagate_retained, stream);
}

SGNSSentenceLearner* SGNSSentenceLearner::deserialize(istream& stream) {
  auto token_learner_(Serializer<SGNSTokenLearner>::deserialize(stream));
  auto ctx_strategy_(Serializer<ContextStrategy>::deserialize(stream));
  auto neg_samples(*Serializer<size_t>::deserialize(stream));
  auto propagate_retained(*Serializer<bool>::deserialize(stream));
  return new SGNSSentenceLearner(
    token_learner_,
    ctx_strategy_,
    neg_samples,
    propagate_retained
  );
}

bool SGNSSentenceLearner::equals(const SGNSSentenceLearner& other) const {
  return
    token_learner->equals(*(other.token_learner)) &&
    ctx_strategy->equals(*(other.ctx_strategy)) &&
    _neg_samples == other._neg_samples &&
    _propagate_retained == other._propagate_retained;
}
*/


/*
//
// SubsamplingSGNSSentenceLearner
//


void SubsamplingSGNSSentenceLearner::sentence_train(const vector<string>&
                                                      words) {
  vector<string> subsampled_words;
  for (auto it = words.begin(); it != words.end(); ++it) {
    const string& word(*it);
    const long word_id = sentence_learner->token_learner->language_model->lookup(word);
    if (word_id < 0 || sentence_learner->token_learner->language_model->subsample(word_id)) {
      subsampled_words.push_back(word);
    } else if (_propagate_discarded) {
      sentence_learner->increment(word);
    }
  }
  sentence_learner->sentence_train(subsampled_words);
}
*/

/*
void SubsamplingSGNSSentenceLearner::serialize(ostream& stream) const {
  Serializer<SGNSSentenceLearner>::serialize(*sentence_learner, stream);
  Serializer<bool>::serialize(_propagate_discarded, stream);
}

SubsamplingSGNSSentenceLearner*
    SubsamplingSGNSSentenceLearner::deserialize(istream& stream) {
  auto sentence_learner_(Serializer<SGNSSentenceLearner>::deserialize(stream));
  auto propagate_discarded(*Serializer<bool>::deserialize(stream));
  return new SubsamplingSGNSSentenceLearner(
    sentence_learner_,
    propagate_discarded
  );
}

bool SubsamplingSGNSSentenceLearner::equals(
    const SubsamplingSGNSSentenceLearner& other) const {
  return
    sentence_learner->equals(*(other.sentence_learner)) &&
    _propagate_discarded == other._propagate_discarded;
}
*/
