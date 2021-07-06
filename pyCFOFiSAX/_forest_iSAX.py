# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import zeros as np_zeros

from pyCFOFiSAX._tree_iSAX import TreeISAX
from tslearn.piecewise import PiecewiseAggregateApproximation


class ForestISAX:
    """
    ForestISAX class containing one or more trees and pretreatment functions on the data contained
    in these trees

    :param int size_word: The size of the SAX words
    :param int threshold: The maximum threshold of nodes
    :param numpy.ndarray data_ts: The sequences to be inserted to extract the stats
    :param int base_cardinality: The smallest cardinality for encoding *i*\ SAX
    :param int number_tree: The number of TreeISAX trees in the forest
    :param list indices_partition: a list of index list where, for each tree, specifies the indices of
    sequences to be inserted
    :param int max_card_alphabet: if ``boolean_card_max == True``, the maximum cardinality of encoding *i*\ SAX in
    each of the trees
    :param boolean boolean_card_max: if ``== True``, Defines a maximum cardinality for encoding *i*\ SAX
    Sequences in each of the trees

    :ivar list length_partition: The length of the SAX words in each tree (``== [size_word]`` if ``number_tree
    == 1``)
    """

    def __init__(self, size_word: int, threshold: int, data_ts: np_ndarray, base_cardinality: int = 2,
                 number_tree: int = 1,
                 indices_partition: list = None,
                 max_card_alphabet: int = 128, boolean_card_max: bool = True):
        """
        Initialization function of the TreeISAX class

        :returns: a forest pointing to one or more iSAX trees
        :rtype: ForestISAX
        """

        # Number of cover contained in the SAX word
        self.size_word = size_word
        # threshold of split node
        self.threshold = threshold
        # Cardinality of each letter at level 1 of the tree
        self.base_cardinality = base_cardinality
        # Max cardinality
        self.max_cardinality = base_cardinality

        self._paa = PiecewiseAggregateApproximation(self.size_word)

        self.forest = {}
        self.number_tree = number_tree

        self.indices_partition = indices_partition

        self._init_trees(data_ts, max_card_alphabet, boolean_card_max)

    def _init_trees(self, data_ts: np_ndarray, max_card_alphabet: int, boolean_card_max: bool):
        """
        Function that initializes the tree (s) when creating a ForestISAX object

        :param numpy.ndarray data_ts: The sequences to be inserted to extract the stats
        :param int max_card_alphabet: if ``boolean_card_max == True``, The maximum cardinality of encoding *i*\ SAX
        dans chacun des arbres
        :param boolean boolean_card_max: if ``boolean_card_max == True``, defines maximum cardinality for encoding *i*\ SAX
        sequences in each tree
        """

        if self.number_tree == 1:
            """ if there is only one tree"""

            self.forest[0] = TreeISAX(
                size_word=self.size_word,
                threshold=self.threshold, data_ts=data_ts,
                base_cardinality=self.base_cardinality,
                max_card_alphabet=max_card_alphabet,
                boolean_card_max=boolean_card_max
            )
            self.length_partition = [self.size_word]
            self.indices_partition = [list(range(self.size_word))]

        elif self.indices_partition is None:
            """ If there is no tree and the indices are not defined """

            self.length_partition = [int(self.size_word / self.number_tree)] * self.number_tree
            for reste in range(self.size_word - sum(self.length_partition)):
                self.length_partition[reste] += 1

            self.indices_partition = []

            for i in range(self.number_tree):
                self.forest[i] = TreeISAX(
                    size_word=self.length_partition[i],
                    threshold=self.threshold,
                    data_ts=data_ts[:, i:self.size_word:self.number_tree],
                    base_cardinality=2,
                    max_card_alphabet=max_card_alphabet,
                    boolean_card_max=boolean_card_max
                )
                self.indices_partition.append(list(range(i, self.size_word, self.number_tree)))

        else:
            # List of letter number in each tree
            self.length_partition = []
            for part_tmp in self.indices_partition:
                self.length_partition.append(len(part_tmp))

            for i in range(self.number_tree):
                self.forest[i] = TreeISAX(
                    size_word=self.length_partition[i],
                    threshold=self.threshold,
                    data_ts=data_ts[:, self.indices_partition[i]],
                    base_cardinality=2,
                    max_card_alphabet=max_card_alphabet,
                    boolean_card_max=boolean_card_max
                )

    def index_data(self, new_sequences: np_ndarray, annotation=None):
        """
        The Index_Data function allows you to insert a large number of sequences

        :param numpy.ndarray new_sequences: The sequences to be inserted
        :param pandas.DataFrame annotation: Pandas dataframe with annotations to be added to each item in a leaf

        :returns: The number of sequences (sub sequences) insert into the tree (in the trees)
        :rtype: numpy.array
        """

        # Ts Conversion to PAA
        if new_sequences.shape[-1] > 1:
            # add dim to avoid tslearn warning
            new_sequences = new_sequences.reshape(new_sequences.shape + (1,))
        npaa = self._paa.fit_transform(new_sequences)

        # To count the number of objects in each tree
        cmpt_insert = np_zeros(shape=self.number_tree)

        for i, tree in self.forest.items():
            # Retrieves the indices of the tree, in the multi-tree case
            npaa_tmp = npaa[:, self.indices_partition[i]]
            npaa_tmp = npaa_tmp.reshape(npaa_tmp.shape[:-1])

            for j,npa_tp in enumerate(npaa_tmp):
                entry = None
                if annotation is not None:
                    entry = annotation.loc[[j],:]
                tree.insert_paa(npa_tp, annotation=entry)
                cmpt_insert[i] += 1

        # Returns array[tree_index] with the number of inserted objects for each tree
        return cmpt_insert

    def _count_nodes(self, id_tree: int):
        """
        The _count_nodes function returns the number of nodes and leaf nodes for a given tree.
        Uses :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.count_nodes_by_tree`.

        :param int id_tree: The tree ID to be analyzed

        :returns: the number of internal nodes, the number of leaf nodes
        :rtype: int, int
        """

        tree = self.forest[id_tree]
        return tree.count_nodes_by_tree()

    def list_nodes(self, id_tree: int, bool_print: bool = False):
        """
        Returns lists of nodes and barycenters of the tree id_tree.Displays statistics on standard output
        if ``bool_print == True``
        Uses :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.get_list_nodes_and_barycentre`.

        :param int id_tree: The tree ID to be analyzed
        :param boolean bool_print: Displays the nodes stats on the standard output

        :returns: The list of nodes, the list of internal nodes, the list of barycenters
        :rtype: list, list, list
        """

        tree = self.forest[id_tree]
        node_list, node_list_leaf, node_leaf_ndarray_mean = tree.get_list_nodes_and_barycentre()
        if bool_print:
            print(f"{len(node_list)} nodes whose {len(node_list_leaf)} leafs in tree {id_tree}")

        return node_list, node_list_leaf, node_leaf_ndarray_mean

    def preprocessing_forest_for_icfof(self, ntss: np_ndarray, bool_print: bool = False, count_num_node: bool = False):
        """
        Allows us to call, for the ``id_tree``  to the pre-treatment for the calculation *i*\ CFOF

        :param ntss: Reference sequences
        :param boolean bool_print: if True, displays the times of each pre-treatment step
        :param boolean count_num_node: if True, count the number of nodes

        :returns: if count_num_node, returns the number of nodes contained in each tree
        :rtype: numpy.array
        """

        total_num_node = np_zeros(self.number_tree)
        for id_tree, tmp_tree in self.forest.items():
            ntss_tmp = ntss[:, self.indices_partition[id_tree]]
            total_num_node[id_tree] = tmp_tree.preprocessing_for_icfof(ntss_tmp,
                                                                       bool_print=bool_print,
                                                                       count_num_node=count_num_node)

        if count_num_node:
            return total_num_node

    def number_nodes_visited(self, query: np_array, ntss: np_ndarray):
        """
        Count the number of average visited nodes in each tree for calculating the approximation.

        :param numpy.array query: The sequence to be evaluated
        :param numpy.ndarray ntss: Reference sequences

        :returns: Returns the number of nodes visited in each tree for the approximation *i*\ CFOF
        :rtype: numpy.array
        """

        total_num_node = np_zeros(self.number_tree*2)

        for id_tree, tmp_tree in self.forest.items():

            sub_query = query[self.indices_partition[id_tree]]
            ntss_tmp = np_array(ntss)[:, self.indices_partition[id_tree]]

            total_num_node[id_tree], total_num_node[self.number_tree + id_tree] = \
                tmp_tree.number_nodes_visited(sub_query, ntss_tmp)

        return total_num_node
