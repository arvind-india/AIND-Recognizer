import warnings
import heapq
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData, beam_size: int=1):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a is word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    for test_word, (X, lengths) in test_set.get_all_Xlengths().items():
        log_likelihoods = {}
        for candidate_word, model in models.items():
            try:
                log_likelihood = model.score(X, lengths)
            except:
                log_likelihood = float('-inf')
            log_likelihoods[candidate_word] = log_likelihood
        probabilities.append(log_likelihoods)

    # todo: Maybe build a table of step-wise highest probability and use it for fast lookups. https://youtu.be/0dVUfYF8ko0?t=74

    def get_best_log_likelihood(sequence=[], sequence_log_likelihood=0, beam_size=3):
        if len(sequence) == len(probabilities):
            return sequence, sequence_log_likelihood
        current_level = probabilities[len(sequence)]
        best_seq, best_log_l = None, float('-inf')

        candidates = heapq.nlargest(beam_size, current_level.items(), lambda x: x[1])
        for word, log_l in candidates:
            new_seq, new_log_l = get_best_log_likelihood(sequence + [word], sequence_log_likelihood + log_l, beam_size)
            if new_log_l > best_log_l:
                best_seq, best_log_l = new_seq, new_log_l
        return best_seq, best_log_l

    guesses, total_log_likelihood = get_best_log_likelihood(beam_size=beam_size)
    return probabilities, guesses
