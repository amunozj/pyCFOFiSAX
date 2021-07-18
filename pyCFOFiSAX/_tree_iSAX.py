# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from pyCFOFiSAX._node import RootNode
from pyCFOFiSAX._isax import IndexableSymbolicAggregateApproximation

from anytree import RenderTree, LevelOrderIter

from scipy.stats import norm
from scipy.spatial.distance import cdist

from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import uint32 as np_uint32
from numpy import float64 as np_float64
from numpy import add as np_add
from numpy import sum as np_sum
from numpy import log2 as np_log2
from numpy import sqrt as np_sqrt
from numpy import square as np_square
from numpy import divide as np_divide
from numpy import mean as np_mean
from numpy import max as np_max
from numpy import min as np_min
from numpy import minimum as np_minimum
from numpy import maximum as np_maximum
from numpy import place as np_place
from numpy import repeat as np_repeat
from numpy import logical_and as np_logical_and
from numpy import less_equal as np_less_equal
from numpy import greater as np_greater
from numpy import logical_not as np_logical_not
from numpy import linspace as np_linspace
from numpy import searchsorted as np_searchsorted
from numpy import multiply as np_multiply
from numpy import round

from scipy.stats import norm as scipy_norm

from bisect import bisect as bisect_bisect

from sys import getsizeof

from time import time as time_time
from sys import stdout

from numba import njit, prange


@njit(nogil=True)
def vrang_seq_ref(distance, max_array, min_array, cdf_mean, cdf_std, num_ts_by_node,
                  index_cdf_bin, cdf_bins):
    """
    Calculates the vrang from the distance between the sequence to be evaluated and the reference sequence.

    :param float distance: The distance between the two sequences
    :param np_array max_array: Max distances between the nodes of the tree and the reference sequence
    :param np_array min_array: MIN distances between the nodes of the tree and the reference sequence
    :param np_array cdf_mean: The average distances between the nodes of the tree and the reference sequence
    :param np_array cdf_std: Dispersion of distances in each leaf node
    :param np_array num_ts_by_node: The number of sequence in each node sheet
    :param np_array index_cdf_bin: l'index de la CDF ``cdf_bins``
    :param np_array cdf_bins: Normal law cdf values centered at the origin and gap-type 1

    :returns: le vrang
    :rtype: int
    """
    vrang = 0

    vrang += num_ts_by_node[
        np_greater(distance, max_array)
    ].sum()

    # tous les nœuds borderlines
    boolean_grp = np_logical_and(np_less_equal(distance, max_array),
                                 np_greater(distance, min_array))

    cdf_mean_grp = cdf_mean[boolean_grp]
    cdf_std_grp = cdf_std[boolean_grp]
    num_ts_by_node_grp = num_ts_by_node[boolean_grp]

    vrang += num_ts_by_node_grp[
        np_logical_and(cdf_std_grp <= 0.0, cdf_mean_grp < distance)
    ].sum()

    cdf_mean_grp = cdf_mean_grp[cdf_std_grp > 0.0]
    num_ts_by_node_grp = num_ts_by_node_grp[cdf_std_grp > 0.0]
    cdf_std_grp = cdf_std_grp[cdf_std_grp > 0.0]

    distance_normalized = (distance - cdf_mean_grp) / cdf_std_grp

    vrang += num_ts_by_node_grp[
        distance_normalized > 4.0
        ].sum()

    new_boolean_grp = np_logical_and(distance_normalized <= 4.0,
                                     distance_normalized >= -4.0)

    num_ts_by_node_grp = num_ts_by_node_grp[new_boolean_grp]
    distance_normalized_grp = distance_normalized[new_boolean_grp]

    index_for_bin = np_searchsorted(index_cdf_bin, distance_normalized_grp)
    vrang += np_multiply(cdf_bins[index_for_bin], num_ts_by_node_grp).sum()
    return vrang


@njit(nogil=True, parallel=True)
def vrang_list_for_all_seq_ref(len_seq_list, distance,
                               max_array, min_array,
                               cdf_mean, cdf_std,
                               num_ts_by_node,
                               index_cdf_bin, cdf_bins):
    """
    Uses the function :func:`~pyCFOFiSAX.tree_iSAX.vrang_seq_ref` For each reference sequence.

    :param float len_seq_list: The number of reference sequence
    :param np_array distance: The distance between the two sequences
    :param np_ndarray max_array: Max distances between the nodes of the tree and the reference sequence
    :param np_ndarray min_array: MIN distances between the nodes of the tree and the reference sequence
    :param np_ndarray cdf_mean: The average distances between the nodes of the tree and the reference sequence
    :param np_array cdf_std: Dispersion of distances in each leaf node
    :param np_array num_ts_by_node: The number of sequence in each node sheet
    :param np_array index_cdf_bin: The index of the CDF ``cdf_bins``
    :param np_array cdf_bins: Normal law cdf values centered at the origin and gap-type 1

    :returns: la liste des vrang
    :rtype: np_array
    """
    vrang_array = np_zeros(len_seq_list)
    for ii_tmp in prange(len_seq_list):
        vrang_array[ii_tmp] = vrang_seq_ref(distance[ii_tmp],
                                            max_array[ii_tmp], min_array[ii_tmp],
                                            cdf_mean[ii_tmp], cdf_std,
                                            num_ts_by_node,
                                            index_cdf_bin, cdf_bins)
    return vrang_array


@njit(nogil=True)
def nodes_visited_for_seq_ref(distance, max_array, min_array, list_parent_node):
    """

    """

    boolean_grp = np_logical_and(np_less_equal(distance, max_array),
                                 np_greater(distance, min_array))
    count_visited_nodes = np_sum(boolean_grp)
    not_boolean_grp = np_logical_not(boolean_grp)
    not_node_parent = list_parent_node[not_boolean_grp]
    boolean_grp = np_logical_and(np_less_equal(distance, max_array[not_node_parent]),
                                 np_greater(distance, min_array[not_node_parent]))
    count_visited_nodes += np_sum(boolean_grp)
    # root is not counted
    count_visited_nodes -= 1
    return count_visited_nodes


@njit(nogil=True, parallel=True)
def nodes_visited_for_all_seq_ref(len_seq_list, distance,
                                  max_array, min_array,
                                  list_parent_node):
    """

    """
    count_visited_nodes = np_zeros(len_seq_list)
    for ii_tmp in prange(len_seq_list):
        count_visited_nodes[ii_tmp] = nodes_visited_for_seq_ref(distance[ii_tmp],
                                                                max_array[ii_tmp], min_array[ii_tmp],
                                                                list_parent_node)
    return count_visited_nodes


class TreeISAX:
    """
    The TreeISAX class contains
     * ISAX distribution of sequences to index
     * and towards the first root node

    .. WARNING::
        In this version, the data_ts are mandatory to define in advance the future breakpoints

    :param int size_word: The number of Sax discretization for each sequence
    :param int threshold: The maximum capacity of the nodes of the tree
    :param numpy.ndarray data_ts: Sequence array to be inserted
    :param int base_cardinality: The smallest cardinality for encoding iSAX
    :param int max_card_alphabet: if self.boolean_card_max == True, Max cardinality for encoding iSAX
    :param float mu: mean value used to define the isax transformation
    :param float sig: standard deviation used to define the isax transformation

    :ivar int size_word: Number of letters contained in the SAX words indexed in the tree
    :ivar int threshold: Threshold before the separation of a leaf into two leaf nodes
    """

    def __init__(self, size_word, threshold, data_ts, base_cardinality=2, max_card_alphabet=128,
                 boolean_card_max=True, mu=None, sig=None):
        """
        Class initialization function TreeISAX

        :returns: a tree iSAX
        :rtype: TreeISAX
        """
        
        # Number of letters contained in the SAX words indexed in the tree
        self.size_word = size_word
        # Threshold before the separation of a sheet into two leaf nodes
        self.threshold = threshold
        # Cardinality of each letter at level 1 of the tree
        # is not very useful ...
        self._base_cardinality = base_cardinality
        # Current Max Cardinality
        self.bigger_current_cardinality = base_cardinality
        # If true, defined maximum cardinality for the alphabet of the tree
        self.boolean_card_max = boolean_card_max
        self.max_card_alphabet = max_card_alphabet

        # mean, variance of data_ts sequences
        self.mu, self.sig = norm.fit(data_ts)

        if mu is not None:
            self.mu = mu

        if sig is not None:
            self.sig = sig

        self.min_max = np_empty(shape=(self.size_word, 2))
        slice_size = int(round(data_ts.shape[1]/self.size_word))
        for i in range(0,self.size_word):            
            if i < self.size_word-1:
                data_slice = data_ts[:, i*slice_size:(i+1)*slice_size]
            else:
                data_slice = data_ts[:, i*slice_size:data_ts.shape[1]]
            self.min_max[i][0] = np_min(data_slice)-self.mu
            self.min_max[i][1] = np_max(data_slice)+self.mu

        self.isax = IndexableSymbolicAggregateApproximation(self.size_word, mean=self.mu, std=self.sig)
        # verif if all values are properly indexable
        tmp_max = np_max(abs(np_array(data_ts) - self.mu))
        bkpt_max = abs(self.isax._card_to_bkpt(self.max_card_alphabet)[-1] - self.mu)
        if tmp_max > bkpt_max:
            ratio = tmp_max / bkpt_max
            self.isax = IndexableSymbolicAggregateApproximation(self.size_word, mean=self.mu, std=self.sig*ratio)
        # and we transmit all this at the root knot
        self.root = RootNode(tree=self, parent=None, sax=[0]*self.size_word,
                             cardinality=np_array([int(self._base_cardinality / 2)] * self.size_word))
        self.num_nodes = 1

        # Attributes for pre-treatment
        self._minmax_nodes_computed = False
        self.node_list = None
        self.node_list_leaf = None
        self.node_leaf_ndarray_mean = None

        # Attributes for calculation *i*\ CFOF
        self._preprocessing_computed = False
        self.min_array = None
        self.max_array = None
        self.min_array_leaf = None
        self.max_array_leaf = None
        self.cdf_mean = None
        self.cdf_std = None

        # Attribute after an update of the tree
        self._new_insertion_after_preproc = False
        self._new_insertion_after_minmax_nodes = False

    def insert(self, new_sequence, annotation=None):
        """
        The converted insert function in PAA then call the function insert_paa

        :param numpy.array new_sequence: The new sequence to be inserted
        :param pandas.DataFrame annotation: annotation to be added to each item in a leaf
        """
        
        # convert to paa
        paa = self.isax.transform_paa([new_sequence])[0]
        self.insert_paa(paa)

    def insert_paa(self, new_paa, annotation=None):
        """
        The insert function that directly calls that of its root node

        :param numpy.array new_paa: The new sequence to be inserted
        :param pandas.DataFrame annotation: annotation to be added to each item in a leaf        
        """

        if len(new_paa) < self.size_word:
            print("Error !! "+new_paa+" is smaller than size.word = "+self.size_word+". FIN")
        else:
            self.root.insert_paa(new_paa, annotation=annotation)
            self._new_insertion_after_preproc = True
            self._new_insertion_after_minmax_nodes = True

    def preprocessing_for_icfof(self, ntss_tmp, bool_print: bool = False, count_num_node: bool = False):
        """
        Allows us to appeal, for the ``id_tree`` tree to the two methods of pre-treatments:
         * :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX._minmax_obj_vs_node`,
         * :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.distrib_nn_for_cdf`.

        :param ntss_tmp: Reference sequences
        :param boolean bool_print: if True, Displays the times of each pre-treatment step
        :param boolean count_num_node: if True, count the number of nodes

        :returns: if ``count_num_node`` True, Returns the number of nodes in the tree
        :rtypes: int
        """

        start_time = time_time()
        self.min_array, self.max_array = self._minmax_obj_vs_node(ntss_tmp, bool_print)
        if bool_print:
            print("_minmax_obj_vs_node --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        start_time = time_time()
        self.distrib_nn_for_cdf(ntss_tmp, bool_print)
        if bool_print:
            print("pretrait cdf --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        start_time = time_time()
        self._minmax_obj_vs_nodeleaf()
        if bool_print:
            print("pretrait _minmax_obj_vs_node_leaf --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        self._preprocessing_computed = True
        if count_num_node:
            return self.num_nodes

    def _minmax_nodes(self):
        """
        Returns the bounds of the nodes of the tree.
        Uses :func:`~pyCFOFiSAX.node.RootNode._do_bkpt`.

        :returns: The min and max bounds of the nodes of the tree
        :rtype: numpy.ndarray
        """

        self.get_list_nodes_and_barycentre()

        bkpt_ndarray = np_ndarray(shape=(2, len(self.node_list), self.size_word), dtype=float)
        for num_n, node in enumerate(self.node_list):
            bkpt_ndarray[0][num_n], bkpt_ndarray[1][num_n], _ = node._do_bkpt()
        return bkpt_ndarray

    def _minmax_obj_vs_node(self, ntss_tmp, bool_print: bool = False):
        """
        Calculates the distances min and max between the sequences ``ntss_tmp`` and the nodes of the tree.

        :param numpy.ndarray ntss_tmp: Reference sequences
        :param boolean bool_print: if True, Displays the times of each pre-treatment step

        :returns: Minimum distances between sequences and nodes, Maximum distances between sequences and nodes
        :rtype: numpy.ndarray, numpy.ndarray
        """

        start_time = time_time()
        bkpt_ndarray = self._minmax_nodes()
        if bool_print:
            print("\t_minmax_nodes --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()
        start_time = time_time()
        row_sax_word, _ = self.isax._row_sax_word_array(ntss_tmp, self.bigger_current_cardinality,
                                                        self.size_word)
        if bool_print:
            print("\t_row_sax_word_array --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        node_list_isax = np_array([node.iSAX_word for node in self.node_list])
        node_list_isax = node_list_isax.transpose((1, 0, 2))

        # TODO card_current = tree.base_cardinality 2/2
        # et donc
        # np_log2(node_list_isax[:,:,1]) - np_log2(tree.base_cardinality)
        node_list_isax[:, :, 1] = np_log2(node_list_isax[:, :, 1])
        node_list_isax[:, :, 1] = node_list_isax[:, :, 1].astype('int64')

        bkpt_ndarray = bkpt_ndarray.transpose(0, 2, 1)
        bkpt_ndarray = bkpt_ndarray.reshape(bkpt_ndarray.shape + (1,))

        tmp_length_node_list = len(self.node_list)
        tmp_length_ntss_tmp = len(ntss_tmp)
        min_array_total = np_zeros((tmp_length_node_list, tmp_length_ntss_tmp), dtype=np_float64)
        max_array_total = np_zeros((tmp_length_node_list, tmp_length_ntss_tmp), dtype=np_float64)

        row_sax_word = row_sax_word.transpose((0, 2, 1))

        ntss_tmp = ntss_tmp.transpose()
        ntss_tmp = ntss_tmp.reshape(ntss_tmp.shape + (1,))

        """ TODO np.vectorize ?"""
        for dim in range(self.size_word):

            for line_node, node_isax in enumerate(node_list_isax[dim]):
                dist_bkptmin = cdist([bkpt_ndarray[0][dim][line_node]], ntss_tmp[dim])[0]
                dist_bkptmax = cdist([bkpt_ndarray[1][dim][line_node]], ntss_tmp[dim])[0]

                distmin = np_minimum(dist_bkptmin, dist_bkptmax)
                max_array_total[line_node] += np_square(np_maximum(dist_bkptmin, dist_bkptmax))

                mask = row_sax_word[dim][node_isax[1]] == node_isax[0]
                np_place(distmin, mask, [0])

                min_array_total[line_node] += np_square(distmin)

        return np_sqrt(min_array_total.transpose()), np_sqrt(max_array_total.transpose())

    def distrib_nn_for_cdf(self, ntss_tmp, bool_print: bool = False):
        """
        Calculates the two indicators, average and standard deviation of the distances, necessary for the use of the CDF of the normal law.
        The calculation of these indicators are described in `Scoring Message Stream Anomalies in Railway Communication Systems, L.Foulon et al., 2019, ICDMWorkshop <https://ieeexplore.ieee.org/abstract/document/8955558>`_.

        :param numpy.ndarray ntss_tmp: Reference sequences
        :param boolean bool_print: and True, Displays the nodes stats on the standard output

        :returns:
        :rtype: list(numpy.ndarray, numpy.array)
        """

        start_time = time_time()
        node_list, node_list_leaf, node_leaf_ndarray_mean = self.get_list_nodes_and_barycentre()
        if bool_print:
            print("pretrait node --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()
            print(len(node_list), " nodes whose ", len(node_list_leaf), " leafs in tree")
            stdout.flush()

        nb_leaf = len(node_list_leaf)

        cdf_mean = np_zeros((nb_leaf, len(ntss_tmp)))
        cdf_std = np_zeros(nb_leaf)
        nb_ts_by_node = np_zeros(nb_leaf, dtype=np_uint32)
        centroid_dist = np_square(cdist(node_leaf_ndarray_mean, ntss_tmp))

        for num, node in enumerate(node_list_leaf):
            cdf_std[node.id_numpy_leaf] = np_mean(node.std)
            nb_ts_by_node[node.id_numpy_leaf] = node.get_nb_sequences()

        dist_list = np_array([np_zeros(i) for i in nb_ts_by_node], dtype=object)

        # calcul distance au carre entre [barycentre et ts] du meme nœud
        """ TODO np.vectorize ?"""
        for node_nn in node_list_leaf:
            dist_list[node_nn.id_numpy_leaf] = cdist([node_nn.mean], node_nn.get_sequences())[0]
        dist_list = np_square(dist_list)

        """ TODO np.vectorize ?"""
        for num, node in enumerate(node_list_leaf):
            node_id = node.id_numpy_leaf

            centroid_dist_tmp = centroid_dist[node_id]
            centroid_dist_tmp = centroid_dist_tmp.reshape(centroid_dist_tmp.shape + (1,))
            centroid_dist_tmp = np_repeat(centroid_dist_tmp, nb_ts_by_node[node_id], axis=1)

            cdf_mean_tmp = np_add(centroid_dist_tmp, dist_list[node_id])
            cdf_mean[node_id] = np_sum(cdf_mean_tmp, axis=1)

        del dist_list
        del cdf_mean_tmp
        del centroid_dist_tmp

        cdf_mean = np_divide(cdf_mean.T, nb_ts_by_node)
        cdf_mean = np_sqrt(cdf_mean)

        self.cdf_mean = cdf_mean
        self.cdf_std = cdf_std

    def _minmax_obj_vs_nodeleaf(self):
        """
        Calculates the min and max distances between the ``ntss_tmp`` sequences and the leaf nodes of the tree.

        .. WARNING::
            Attention must be executed after :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX._minmax_obj_vs_node` and :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.distrib_nn_for_cdf`.
        """
        id_leaf_array = np_zeros(len(self.node_list_leaf), dtype=np_uint32)
        for i_tmp in range(len(self.node_list_leaf)):
            id_leaf_array[i_tmp] = self.node_list_leaf[i_tmp].id_numpy
        self.min_array_leaf = self.min_array[:, id_leaf_array]
        self.max_array_leaf = self.max_array[:, id_leaf_array]

    def get_level_max(self):
        """
        Function to return the max level Switching root level 0

        :returns: The max depth level
        :rtype: int
        """

        lvl_max = 0
        for _, _, node in RenderTree(self.root):
            if node.level > lvl_max: lvl_max = node.level
        return lvl_max

    def get_width_of_level(self, level: int):
        """
        Function to return the width of a root level level level 0

        :returns: the number of node on the level of the tree
        :rtype: int
        """

        cmpt = 0
        for _, _, node in RenderTree(self.root):
            if node.level == level: cmpt += 1
        return cmpt

    def get_width_of_all_level(self):
        """
        Function to return the width of all levels in a list that knows root level 0

        :returns: The number of node on each level of the tree
        :rtype: list
        """

        cmpt = []
        for _, _, node in RenderTree(self.root):
            while node.level > len(cmpt)-1:
                cmpt.append(0)
            cmpt[node.level] += 1
        return cmpt

    def get_nodes_of_level(self, level: int):
        """
        Function to return the nodes of a level knowing root level 0

        :param int level: The level of the tree to evaluate

        :returns: The nodes of the level-ie level of the tree
        :rtype: list
        """
        
        node_list = []
        for _, _, node in RenderTree(self.root):
            if node.level == level:
                node_list.append(node)
        return node_list

    def get_nodes_of_level_or_terminal(self, level: int):
        """
        Function to return the nodes of a level knowing root level 0 or terminal nodes above that level

        :param int level: The level of the tree to evaluate

        :returns: The nodes of the level-ie level of the tree
        :rtype: list
        """
        
        node_list = []
        for node in LevelOrderIter(self.root, maxlevel=level):
            if node.level == level:
                node_list.append(node)
            elif node.level < level and node.terminal:
                node_list.append(node)
        return node_list        

    def get_number_internal_and_terminal(self):
        """
        Function to return the number of leaf nodes and internal nodes

        :returns: the number of internal nodes, the number of leaves nodes
        :rtype: int, int
        """

        cmpt_leaf = 0
        cmpt_internal = 0
        for _, _, node in RenderTree(self.root):
            if node.terminal:
                cmpt_leaf += 1
            else:
                cmpt_internal += 1
        return cmpt_internal, cmpt_leaf

    def count_nodes_by_tree(self):
        """
        The COUNT_NODES_BY_TREE function returns the number of nodes and sheet nodes of the shaft.
        Uses :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.get_number_internal_and_terminal`.

        :returns: the number of internal nodes, the number of leaves nodes
        :rtype: int, int
        """

        cmpt_internal, cmpt_leaf = self.get_number_internal_and_terminal()
        cmpt_node = cmpt_leaf + cmpt_internal
        return cmpt_node, cmpt_leaf

    def get_list_nodes_and_barycentre(self):
        """
        Returns Lists of Nodes and Barketters

        :returns: List of nodes, List of Leaves Nodes, List of Leave Barycenters
        :rtype: list, list, list
        """

        if self._minmax_nodes_computed and not self._new_insertion_after_minmax_nodes:
            return self.node_list, self.node_list_leaf, self.node_leaf_ndarray_mean

        self.node_list = []
        self.node_list_leaf = []
        node_leaf_mean = []
        cmpt_node = 0
        count_leaf = 0

        for _, _, node in RenderTree(self.root):
            self. node_list.append(node)
            node.id_numpy = cmpt_node
            cmpt_node += 1
            if node.is_leaf:
                self.node_list_leaf.append(node)
                node_leaf_mean.append(node.mean)
                node.id_numpy_leaf = count_leaf
                count_leaf += 1

        if cmpt_node != self.num_nodes:
            print("ATTENTION !!!")
            exit(-1)
        self.node_leaf_ndarray_mean = np_array(node_leaf_mean)
        self._minmax_nodes_computed = True
        self._new_insertion_after_minmax_nodes = False
        return self.node_list, self.node_list_leaf, self.node_leaf_ndarray_mean

    def get_list_nodes_leaf(self):
        """
        Returns List of Leaves Nodes

        :returns: List of leaves nodes
        :rtype: list
        """
        self.get_list_nodes_and_barycentre()
        return self.node_list_leaf

    def get_size(self):
        """
        Function to return the memory size of the shaft, nodes
        and sequences contained in the tree

        :returns: Total memory size, nodes' memory size, memory size of the sequences
        :rtype: int, int, int
        """

        total_size = 0
        ts_size = 0
        node_size = 0
        total_size += getsizeof(self)
        for _, _, node in RenderTree(self.root):
            total_size += getsizeof(node)
            node_size += getsizeof(node)
            if node.terminal:
                for ts in node.get_sequences():
                    total_size += getsizeof(ts)
                    ts_size += getsizeof(ts)
        return total_size, node_size, ts_size

    def get_size_and_width_and_number_types_nodes(self):
        """
        Feature grouping:
            -  get_size()
            -  get_width_of_all_level()
            -  get_number_internal_and_terminal()

        :returns: Total memory size, memory size of nodes, memory size of the sequences,
            the number of nodes on each level, the number of internal nodes, the number of sheet nodes,
            and the number of sequence inserted in the tree
        :rtype: int, int, int, list, int, int, int
        """

        total_size = 0
        node_size = 0
        ts_size = 0

        cmpt_nodes = []

        number_terminal = 0
        number_internal = 0

        cmpt_sequences = []

        total_size += getsizeof(self)
        total_size += getsizeof(self.root)
        for _, _, node in RenderTree(self.root):
            while node.level > len(cmpt_nodes)-1:
                cmpt_nodes.append(0)
                cmpt_sequences.append(0)
            cmpt_nodes[node.level] += 1
            if node.terminal:
                number_terminal += 1
                cmpt_sequences[node.level] += len(node.get_sequences())
            else:
                number_internal += 1
            total_size += getsizeof(node)
            node_size += getsizeof(node)
            if node.terminal:
                for ts in node.get_sequences():
                    total_size += getsizeof(ts)
                    ts_size += getsizeof(ts)
        return total_size, node_size, ts_size, cmpt_nodes, number_internal, number_terminal, cmpt_sequences

    def vrang_list_faster(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Get the vrang list for the ``sub_query`` sequence in the tree.
        Necessary for the calculation of the approximation.
        This method is the fast version of the method :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.vrang_list`.

        .. note::
            This method does not travel the tree, but directly prunes the leaves nodes.
            Preserved (uncut) leaves will be used by the approximation function.

        :param sub_query: The sequence to be evaluated
        :param ntss_tmp: Reference sequences (IE. Reference history) in PAA format

        :returns: The vrang list of``sub_query``
        :rtype: list(float)
        """

        num_ts_by_node = []
        for i, node in enumerate(self.get_list_nodes_leaf()):
            num_ts_by_node.append(node.get_nb_sequences())
        num_ts_by_node = np_array(num_ts_by_node)

        if not hasattr(self, 'index_cdf_bin'):
            self.index_cdf_bin = np_linspace(-4.0, 4.0, num=1000)
        if not hasattr(self, 'cdf_bins'):
            self.cdf_bins = scipy_norm.cdf(self.index_cdf_bin, 0, 1)

        q_paa = self.isax.transform_paa([sub_query])[0]

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])], ntss_tmp)[0]

        return vrang_list_for_all_seq_ref(len(ntss_tmp), distance_q_p,
                                          self.max_array_leaf, self.min_array_leaf,
                                          self.cdf_mean, self.cdf_std,
                                          num_ts_by_node,
                                          self.index_cdf_bin, self.cdf_bins)

    def vrang_list(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Get the vrang list for the ``sub_query`` sequence in the tree.
        Necessary for the calculation of the approximation.
        The same method faster but without the tree course:
        :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.vrang_list_faster`.

        :param sub_query: The sequence to be evaluated
        :param ntss_tmp: Reference sequences (IE. Reference history)

        :returns: The vrang list of``sub_query``
        :rtype: list(float)
        """

        if not hasattr(self, 'index_cdf_bin'):
            self.index_cdf_bin = np_linspace(-4.0, 4.0, num=1000)
        if not hasattr(self, 'cdf_bins'):
            self.cdf_bins = scipy_norm.cdf(self.index_cdf_bin, 0, 1)

        # List of vrang
        # TODO np_array
        k_list_result = []

        q_paa = self.isax.transform_paa([sub_query])[0]

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])], ntss_tmp)[0]

        # For any object P
        for stop_ite, p_paa in enumerate(ntss_tmp):

            p_name = stop_ite

            # numbers of objects too close
            count_ts_too_nn = 0

            # Real distance
            distance = distance_q_p[stop_ite]

            max_array_p_bool = self.max_array[p_name] < distance
            min_array_p_bool = self.min_array[p_name] <= distance

            nodes_list_fifo = []
            nodes_list_fifo.extend(self.root.nodes)

            while nodes_list_fifo:

                node_nn = nodes_list_fifo.pop(0)

                if max_array_p_bool[node_nn.id_numpy]:

                    count_ts_too_nn += node_nn.get_nb_sequences()

                elif node_nn.terminal:

                    if min_array_p_bool[node_nn.id_numpy]:

                        cdf_mean_tmp = self.cdf_mean[p_name][node_nn.id_numpy_leaf]
                        cdf_std_tmp = self.cdf_std[node_nn.id_numpy_leaf]

                        if cdf_std_tmp > 0.0:
                            distance_normalized = (distance - cdf_mean_tmp) / cdf_std_tmp
                            if distance_normalized > 4:
                                count_ts_too_nn += node_nn.get_nb_sequences()
                            elif -4 <= distance_normalized:
                                index_for_bin = bisect_bisect(self.index_cdf_bin, distance_normalized)
                                count_ts_too_nn += self.cdf_bins[index_for_bin] * node_nn.get_nb_sequences()
                        else:
                            if distance > cdf_mean_tmp:
                                count_ts_too_nn += node_nn.get_nb_sequences()

                elif min_array_p_bool[node_nn.id_numpy]:
                    nodes_list_fifo.extend(node_nn.nodes)

            # The estimation of the position of the Centroid of Query is saved compared to P
            k_list_result.append(count_ts_too_nn)

        return k_list_result

    def number_nodes_visited(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Account the number of average visited nodes in the tree for calculating the approximation.

        :param numpy.array sub_query: The sequence to be evaluated
        :param numpy.ndarray ntss_tmp: Reference sequences

        :returns: Returns the number of nodes visited in the tree for the approximation *i*\ CFOF
        :rtype: numpy.array
        """
        q_paa = self.isax.transform_paa([sub_query])[0]
        ntss_tmp_paa = self.isax.transform_paa(ntss_tmp)

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])],
                             ntss_tmp_paa.reshape(ntss_tmp_paa.shape[:-1]))[0]

        list_parent_node = np_zeros(len(self.node_list), dtype=np_uint32)

        for tmp_node in self.node_list:
            if tmp_node.id_numpy == 0:
                continue
            list_parent_node[tmp_node.id_numpy] = tmp_node.parent.id_numpy

        count_visited_nodes_list = nodes_visited_for_all_seq_ref(len(ntss_tmp_paa), distance_q_p,
                                                                 self.max_array, self.min_array,
                                                                 list_parent_node)

        return self.num_nodes, count_visited_nodes_list.mean()
