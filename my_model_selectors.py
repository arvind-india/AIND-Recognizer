import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        try:
            best_bic, best_model = float("inf"), 1
            for num_components in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(num_components)  # type: GaussianHMM
                log_likelihood = model.score(self.X, self.lengths)

                # Number of parameters
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/8?u=markus-131
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12?u=markus-131
                num_samples, num_features = self.X.shape

                # The number of initial state probabilities is simply the number of possible states ("components")
                num_initial_state_probabilities = num_components

                # Each state is only connected to itself and its immediate follower; no skip connections
                # are allowed, the graph is otherwise acyclic.
                # The number of free parameters for the transition probabilities is then determined as follows:
                # Since the probabilities for leaving a node must sum to 1, the probability of moving
                # to the follower is directly determined by the probability to stay, resulting in N
                # parameters for the N states. However, the last state has no follower, thus the number
                # of parameters is reduced by one.
                # If the model would be allowed to "skip" over connections, essentially allowing connections
                # to any state (even earlier ones), the number of parameters would be N*(N-1), determined
                # by (N-1) free transition probabilities for each of the N states.
                # If no backwards connections are allowed (the lower left triangle of the transition matrix is zero)
                # the number of free parameters would be N*(N-1)/2.
                num_transition_probabilities = num_components - 1

                # The number of emission probabilities is the number of means and covariances.
                # Since the matrix is configured to be diagonal in the GaussianHMM constructor,
                # there only is the squared variance.
                num_emission_probabilities = num_components * num_features * 2

                num_params = num_initial_state_probabilities + num_transition_probabilities + num_emission_probabilities
                bic = -2 * log_likelihood + num_params * np.log(num_samples)

                # Occam's razor: We try smaller models first and check for a
                # strictly smaller BIC. This will prefer smaller models in case of a draw.
                if bic < best_bic:
                    best_bic, best_model = bic, model
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        try:
            best_dic, best_model = float("-inf"), None
            for num_components in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(num_components)  # type: GaussianHMM

                log_likelihood = model.score(self.X, self.lengths)
                log_likelihoods = [model.score(X, lengths)
                                   for word, (X, lengths)
                                   in self.hwords.items()
                                   if word != self.this_word]

                dic = log_likelihood - np.sum(log_likelihoods) / (len(log_likelihoods) - 1)

                if dic > best_dic:
                    best_dic, best_model = dic, model
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            highest_log_likelihood, best_p = float("-inf"), 1

            split_method = KFold(min(3, len(self.sequences)))
            for p in range(self.min_n_components, self.max_n_components + 1):
                log_likelihoods = []

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    model = self.base_model(p)  # type: GaussianHMM

                    X, lengths = combine_sequences(cv_test_idx, self.sequences)
                    log_likelihoods.append(model.score(X, lengths))

                log_likelihood = np.mean(log_likelihoods)
                if log_likelihood > highest_log_likelihood:
                    highest_log_likelihood, best_p = log_likelihood, p

            return self.base_model(best_p)
        except:
            return self.base_model(self.n_constant)
