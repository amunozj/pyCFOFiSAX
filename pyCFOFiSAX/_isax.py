# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from numpy import int as np_int
from numpy import array as np_array
from numpy import zeros as np_zeros

from math import log as math_log

from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise.piecewise import _paa_to_symbols, _breakpoints, _bin_medians
from tslearn.utils import to_time_series_dataset


class IndexableSymbolicAggregateApproximation(PiecewiseAggregateApproximation):
    """Indexable Symbolic Aggregate approXimation (iSAX) transformation.

    First presented by J. Shieh & E. Keogh in *i*\ SAX: Indexing and Mining Terabyte Sized Time Series.
    Class that inherits the class ``PiecewiseAggregateApproximation`` proposed by Romain Tavenard in \`\`tslearn\`\` Available here <https://tslearn.readthedocs.io/en/stable/>`_.

    :param int n_segments: The number of letters in the word sax
    :param int alphabet_size_min: The minimum size of the Sax alphabet at initialization (2 default)
    :param float mean: The average of the distribution of encoder sequences (0.0 default)
    :param float std: The standard deviation of the distribution of encoder sequences (0.0 default)
    """

    def __init__(self, n_segments, alphabet_size_min=2, mean=0.0, std=1.0):
        """
        Initialization function of the class IndexableSymbolicAggregateApproximation

        :returns: a class of encoding *i*\ SAX
        :rtype: IndexableSymbolicAggregateApproximation
        """

        PiecewiseAggregateApproximation.__init__(self, n_segments)
        self.n_segments = n_segments
        self.alphabet_size_min = alphabet_size_min
        self.alphabet_size_max = alphabet_size_min

        self.mean = mean
        self.std = std

        self.card_to_bkpt_ = dict()
        self.card_to_bkpt_only_ = dict()
        self.card_to_bkpt_middle_ = dict()
        self.card_to_bkpt_[self.alphabet_size_min] = _breakpoints(self.alphabet_size_min, scale=self.std) + self.mean
        self.card_to_bkpt_only_[self.alphabet_size_min] = _breakpoints(self.alphabet_size_min,
                                                                       scale=self.std) + self.mean
        self.card_to_bkpt_middle_[self.alphabet_size_min] = _bin_medians(self.alphabet_size_min,
                                                                         scale=self.std) + self.mean

    def fit(self, X):
        """
        Prepares the data for encoding *i*\ SAX according to``PiecewiseAggregateApproximation``

        :returns: Received data for encoding, defined by ``tslearn``
        :rtype: numpy.ndarray of PiecewiseAggregateApproximation
        """

        return PiecewiseAggregateApproximation.fit(self, X)

    def _card_to_bkpt(self, max_cardinality):
        """
        Returns the breakpoints associated with the cardinality <= max_cardinality.
        The function calculates and stores the BKPT if they have never been calculated.

        :param int max_cardinality: Maximum cardinality

        :returns: Breakpoints associated with cardinality <= max_cardinality
        :rtype: dict
        """

        if max_cardinality not in self.card_to_bkpt_:

            self.card_to_bkpt_[max_cardinality] = _breakpoints(max_cardinality, scale=self.std) + self.mean
            self.card_to_bkpt_middle_[max_cardinality] = _bin_medians(max_cardinality, scale=self.std) + self.mean

        if max_cardinality > self.alphabet_size_max:
            self.alphabet_size_max = max_cardinality

        return self.card_to_bkpt_[max_cardinality]

    def _card_to_bkpt_only(self, max_cardinality):
        """
        Returns the breakpoints associated with cardinality == max_cardinality.
        The function calculates and stores the BKPs if they have never been calculated.

        :param int max_cardinality: cardinality

        :returns: Breakpoints associated with cardinality == max_cardinality
        :rtype: list
        """

        if max_cardinality not in self.card_to_bkpt_only_:

            bkpt_up = self._card_to_bkpt(max_cardinality)
            bkpt_low = self._card_to_bkpt(max_cardinality/2)

            self.card_to_bkpt_only_[max_cardinality] = [i for i in bkpt_up if i not in bkpt_low]

        return self.card_to_bkpt_only_[max_cardinality]

    def fit_transform(self, X, card, **fit_params):
        """
        Prepares the ``X`` data provided in parameter for encoding``tslearn``.
        Then transforms the ``X`` data provided as a parameter first in PAA and then in cardinate cardinality ``card``.

        :param numpy.ndarray X: Data to transform
        :param int card: Cardinality to use for processing

        :returns: data transformed into SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._fit(X_)._transform(X_, card)

    def _transform_paa_to_isax(self, X_paa, card):
        """
        Transforms ``X_paa`` data into *i*\ SAX parameters according to cardinality ``card``.

        :param numpy.ndarray X_paa: PAA data to transform into *i*\ SAX
        :param list card: Cardinalities to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """
        
        max_card = max(card)
        isax_list = _paa_to_symbols(X_paa, self._card_to_bkpt(max_card))
        return np_array([[(symbol[0]*card[idx_sym] / max_card,
                           card[idx_sym]) for idx_sym, symbol in enumerate(word)] for word in isax_list], dtype=np_int)

    def _transform(self, X, card):
        """
        Transforms ``X`` data in parameter first into PAA and then in cardinate cardinality ``card``.

        :param numpy.ndarray X: Data to transform
        :param int card: Cardinality to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """
        
        X_paa = PiecewiseAggregateApproximation._transform(self, X)
        return self._transform_paa_to_isax(X_paa, card)

    def transform_paa_to_isax(self, X_paa, card):
        """
        Prepares ``X_paa`` data provided as a parameter for encoding``tslearn``.
        Then transforms ``X_paa`` data into *i*\ SAX parameter according to cardinalities ``card``.

        :param numpy.ndarray X_paa: PAA data to transform into *i*\ SAX
        :param list card: Cardinalities to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """

        X_paa_ = to_time_series_dataset(X_paa)
        return self._transform_paa_to_isax(X_paa_, card)

    def transform(self, X, card):
        """
        Prepares the ``X`` data provided in parameter for encoding ``tslearn``.
        Then transforms ``X`` data in parameter first into PAA and then in cardinatlity of cardinality ``card``.

        :param numpy.ndarray X: Data to transform
        :param int card: Cardinality to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._transform(X_, card)

    def _transform_sax(self, X, card):
        """
        Transforms ``X`` data in parameter first into PAA and then in cardinate cardinality ``card``.

        :param numpy.ndarray X: Data to transform
        :param int card: Cardinality to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """
        
        X_paa = PiecewiseAggregateApproximation._transform(self, X)
        # isax_list = _paa_to_symbols(X_paa, self._card_to_bkpt(card))
        return _paa_to_symbols(X_paa, self._card_to_bkpt(card))

    def transform_sax(self, X, card):
        """
        Prepares the ``X`` data provided in parameter for encoding``tslearn``.
        Then transforms ``X`` data in parameter first into PAA and then in cardinate cardinality``card``.

        :param numpy.ndarray X: Data to transform
        :param int card: Cardinality to use for processing

        :returns: Transformed data in SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._transform_sax(X_, card)

    def transform_paa(self, X):
        """
        Prepares the ``X`` data provided in parameter for encoding``tslearn``.
        Then transforms ``X`` data into parameter in PAA.

        :param numpy.ndarray X: Data to transform

        :returns: Transformed data in PAA
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return PiecewiseAggregateApproximation._transform(self, X_)

    def _row_sax_word_array(self, ntss_tmp, bigger_cardinality, size_word):
        """
        Convert all sequences according to the different cardinality of the tree.
        For each cardinality, uses :func:`~pyCFOFiSAX.isax.IndexableSymbolicAggregateApproximation.transform_sax`.

        :param ntss_tmp: The sequences to be analyzed
        :param int bigger_cardinality: The greatest cardinality *i*\ SAX of the tree
        :param int size_word: The size of the SAX sequences of the tree

        :returns: SAX words from all ``ntss_tmp`` sequences according to all the cardinalities of the tree, a dict returning the cardinality index *i*\ SAX
        :rtype: numpy.ndarray, dict
        """

        # TODO integrate tree.base_cardinality in the calculation
        # si tree.base_cardinality = 3 par exemple...
        number_of_card = int(math_log(bigger_cardinality, 2))

        row_sax_word = np_zeros((number_of_card + 1, len(ntss_tmp), size_word, 1))
        # TODO card_current = tree.base_cardinality 1/2
        card_current = 1
        card_index = 0
        card_to_index = dict()
        """ TODO np.vectorize
        >>> vtransform_sax = np.vectorize(tree.isax.transform_sax)
        >>> card_list = 2**np.arange(1,int(np.sqrt(tree.bigger_current_cardinality))-1)
        >>> vtransform_sax(ntss_tmp, card_list)"""
        while card_current <= bigger_cardinality:
            card_to_index[card_current] = card_index
            row_sax_word[card_index] = self.transform_sax(ntss_tmp, card_current)
            card_current = card_current * 2
            card_index += 1
        row_sax_word = row_sax_word.transpose((2, 1, 0, 3))

        return row_sax_word.reshape(row_sax_word.shape[:-1]), card_to_index
