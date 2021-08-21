# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause
from collections import defaultdict

from anytree import Node
from numpy import all as np_all
from numpy import copy as np_copy
from numpy import mean as np_mean
from numpy import std as np_std
from numpy import sqrt as np_sqrt
from numpy import zeros as np_zeros
from numpy import array as np_array
from numpy import empty as np_empty
from numpy import argmin as np_argmin
import pandas as pd
import numpy as np

""" Module for grouping the three types of nodes used by the Isax tree """


class RootNode(Node):
    """
    The RootNode class creates the only node of the ancestor tree common to all other nodes

    :param tree_iSAX tree: the tree in which the node is contained
    :param Node parent: The parent parent node
    :param numpy.array sax: SAX values of the node
    :param numpy.array cardinality: Cardinality of SAX values
    """

    #: Attribute to define an ID for each node
    id_global = 0

    def __init__(self, tree, parent, sax, cardinality):
        """
        Initialization function of the rootnode class

        :returns: a root node
        :rtype: RootNode
        """

        self.iSAX_word_small = ''
        for i in range(0,len(sax)):
            if i!=0:
                self.iSAX_word_small += '_'

            self.iSAX_word_small = self.iSAX_word_small + str(sax[i]) + '.' + str(cardinality[i])

        self.iSAX_word = np_array([sax, cardinality]).T

        Node.__init__(self, parent=parent, name=str(self.iSAX_word))

        self.tree = tree
        self.sax = sax
        self.cardinality = cardinality
        self.short_name = self.iSAX_word_small

        self.cardinality_next = np_copy(self.cardinality)
        self.cardinality_next = np_array([x*2 for x in self.cardinality_next])

        # Number of sequences contained in the node (or by its sons)
        self.nb_sequences = 0

        """ The incremental computing part for CFOF """
        self.mean = np_empty(shape=self.tree.size_word)
        # Allows the incremental calculation of self.mean
        self.sum = np_empty(shape=self.tree.size_word)

        self.std = np_empty(shape=self.tree.size_word)
        # Allows the incremental calculation of self.std
        self.sn = np_empty(shape=self.tree.size_word)

        # Specific to internal nodes
        self.nodes = []
        self.key_nodes = {}

        self.terminal = False
        self.level = 0

        self.id = RootNode.id_global
        RootNode.id_global += 1

    def insert_paa(self, new_paa, annotation=None, parallel=False):
        """
        The insert_paa(new_paa) function to insert a new converted sequence into PAA

        :param new_paa: The converted sequence in PAA to insert
        :param dictionary annotation: Dictionary of lists with annotations to be added to each item in a leaf
        :param boolean parallel: if True, it runs the indexing of the tree at minimum cardinality and waits for the parallel escalation      
        """

        i_sax_word = self.tree.isax.transform_paa_to_isax(new_paa, self.cardinality_next)[0]
        # for i_sax_word, we return the first element of each tuple and we test if the word appears in the nodes
        if str([i[0] for i in i_sax_word]) in self.key_nodes:
            # We recover the node that sticks to the word
            current_node = self.key_nodes[str([i[0] for i in i_sax_word])]

            # If it's a leaf
            if current_node.terminal:
                # and that we do not exceed the max threshold, or the leaf node is no longer splitable,
                # or we are pre-sorting at minmimum cardinality for parallelization
                # nb : This second condition is not suggested by Shieh and Kheogh
                if current_node.nb_sequences < self.tree.threshold or not current_node.splitable or parallel:
                    current_node.insert_paa(new_paa, annotation=annotation)
                # But otherwise (we exceed the max threshold and the leaf is splitable)
                else:
                    # Creation of the new internal node
                    new_node = InternalNode(self.tree, current_node.parent, np_copy(current_node.sax),
                                            np_copy(current_node.cardinality), current_node.sequences)
                    # We insert the new sequence in this new internal node
                    new_node.insert_paa(new_paa, annotation=annotation, parallel=parallel)
                    # For each of the sequences of the current leaf are inserted its sequences in the new internal node
                    # This internal node will create one or more leaves to insert these sequences
                    for j,ts in enumerate(current_node.sequences):

                        entry = {}
                        for key, value in current_node.annotations.items():
                            entry[key] = value[j]
                            
                        new_node.insert_paa(ts, entry, parallel=parallel)
                        
                    # and we delete the current leaf from the list of nodes
                    self.nodes.remove(current_node)
                    # that we also remove from Dict
                    del self.key_nodes[str(current_node.sax)]
                    # and we add to the dict the new internal node
                    self.key_nodes[str(current_node.sax)] = new_node
                    self.nodes.append(new_node)
                    current_node.parent = None
                    # and we definitely delete the current leaf
                    del current_node

            # Otherwise (it's not a leaf) we continue the search of the tree
            else:
                current_node.insert_paa(new_paa, annotation=annotation, parallel=parallel)

        # Otherwise (the Sax node does not exist) we create a new leaf
        else:
            new_node = TerminalNode(self.tree, self, [i[0] for i in i_sax_word], np_array(self.cardinality_next))
            new_node.insert_paa(new_paa, annotation=annotation)
            self.key_nodes[str([i[0] for i in i_sax_word])] = new_node
            self.nodes.append(new_node)
            self.tree.num_nodes += 1

        # Shift of node indicators
        self.nb_sequences += 1
        # calculate mean and std
        if self.nb_sequences == 1:
            self.sum = np_copy(new_paa)
            self.mean = np_copy(new_paa)
            self.std = np_zeros(self.tree.size_word)
            self.sn = np_zeros(self.tree.size_word)
        else:
            mean_moins_1 = np_copy(self.mean)
            self.sum += new_paa
            self.mean = self.sum / self.nb_sequences
            self.sn += (new_paa - mean_moins_1) * (new_paa - self.mean)
            self.std = np_sqrt(self.sn / self.nb_sequences)

    def escalate(self, node):
        """
        A function that triggers a full cardinality escalation underneath a node

        :param node: The node to escalate
        """

        # Creation of the new internal node
        new_node = InternalNode(self.tree, node.parent, np_copy(node.sax),
                                np_copy(node.cardinality), node.sequences)

        # For each of the sequences of the current leaf are inserted its sequences in the new internal node
        # This internal node will create one or more leaves to insert these sequences
        for j,ts in enumerate(node.sequences):
            entry = {}
            for key, value in node.annotations.items():
                entry[key] = value[j]
                
            new_node.insert_paa(ts, entry)

        # and we delete the current leaf from the list of nodes
        self.nodes.remove(node)
        # that we also remove from Dict
        del self.key_nodes[str(node.sax)]
        # and we add to the dict the new internal node
        self.key_nodes[str(node.sax)] = new_node
        self.nodes.append(new_node)
        node.parent = None
        # and we definitely delete the current leaf
        del node 

    def _do_bkpt(self):
        """
        The _do_bkpt function calculates the min and max bounds of the node on each dimension of the node.
        as well a the expected value of each node

        :returns: an array containing the min bounds, one containing the max bounds, and one containing the
        expected value
        :rtype: numpy.array, numpy.array, numpy.array
        """

        bkpt_list_min = np_empty(self.tree.size_word)
        bkpt_list_max = np_empty(self.tree.size_word)
        bkpt_list_exp = np_empty(self.tree.size_word)
        for i, iSAX_letter in enumerate(self.iSAX_word):
            bkpt_tmp = self.tree.isax._card_to_bkpt(iSAX_letter[1])
            # The case where there is no BKPT (root node)
            if iSAX_letter[1] < 2:
                bkpt_list_min[i] = self.tree.min_max[i][0]
                bkpt_list_max[i] = self.tree.min_max[i][1]
            # the case where there is no lower bound BKPT 
            elif iSAX_letter[0] == 0:
                bkpt_list_min[i] = self.tree.min_max[i][0]
                bkpt_list_max[i] = bkpt_tmp[iSAX_letter[0]]
            # the case where there is no upper bound BKPT
            elif iSAX_letter[0] == iSAX_letter[1]-1:
                bkpt_list_min[i] = bkpt_tmp[iSAX_letter[0]-1]
                bkpt_list_max[i] = self.tree.min_max[i][1]
            # The general case
            else:
                bkpt_list_min[i] = bkpt_tmp[iSAX_letter[0]-1]
                bkpt_list_max[i] = bkpt_tmp[iSAX_letter[0]]
            # Escalate cardinality by one and find expected value
            bkpt_tmp = self.tree.isax._card_to_bkpt(iSAX_letter[1]*2)
            bkpt_list_exp[i] = bkpt_tmp[np.logical_and(bkpt_tmp>bkpt_list_min[i], bkpt_tmp<bkpt_list_max[i])]

        return bkpt_list_min, bkpt_list_max, bkpt_list_exp

    def get_min_max_distance(self, node2, metric = 'L2'):

        """
        Function that calculates min, max, norm, and expected distances between two nodes.

        :returns: an array containing the three distances
        :rtype: numpy.array
        """        

        node1_bkpt = self._do_bkpt()
        node2_bkpt = node2._do_bkpt()

        distance = np.array([np.abs(node1_bkpt[0] - node2_bkpt[0]), np.abs(node1_bkpt[1] - node2_bkpt[1])
                             , np.abs(node1_bkpt[0] - node2_bkpt[1]) , np.abs(node1_bkpt[1] - node2_bkpt[0])])

        mu = self.tree.isax.mean

        node1_norm_bkpt = node1_bkpt[1].copy()
        node1_norm_bkpt[node1_bkpt[1]>mu] = node1_bkpt[0][node1_bkpt[1]>mu]

        node2_norm_bkpt = node2_bkpt[1].copy()
        node2_norm_bkpt[node2_bkpt[1]>mu] = node2_bkpt[0][node2_bkpt[1]>mu]

        norm_dist = np.abs(node1_norm_bkpt-node2_norm_bkpt)
        expect_dist = np.abs(node1_bkpt[2]-node2_bkpt[2])

        if metric == 'L1':
            min_dist = np.sum(np.min(distance, axis=0))
            max_dist = np.sum(np.max(distance, axis=0))
            norm_dist = np.sum(norm_dist)
            expect_dist = np.sum(expect_dist)
        else:
            min_dist = np.sqrt(np.sum(np.min(distance, axis=0)**2))
            max_dist = np.sqrt(np.sum(np.max(distance, axis=0)**2))
            norm_dist = np.sqrt(np.sum(norm_dist**2))
            expect_dist = np.sqrt(np.sum(expect_dist**2))


        

        return np.array([min_dist, max_dist, norm_dist, expect_dist])

    def get_annotations(self):
        """
        Returns the annotations contained in the node (leaf only) or its descendants

        :returns: Annotations
        :rtype: numpy.ndarray
        """

        
        annotations = defaultdict(list)
        for node in self.nodes:

            node_annotations = node.get_annotations()
            for key, value in node_annotations.items():
                annotations[key].append(value)

        return annotations

    def get_sequences(self):
        """
        Returns the sequences contained in the node (leaf only) or its descendants

        :returns: Sequences
        :rtype: numpy.ndarray
        """
        sequences = []
        for node in self.nodes:
            for ts in node.get_sequences():
                sequences.append(ts)
        return sequences

    def get_nb_sequences(self) -> int:
        """
        Returns the number of sequences contained in the node and its descendants

        :returns: The number of sequences of the subtree
        :rtype: int
        """
        return self.nb_sequences

    def __str__(self):
        """
        Setting the display function for the node

        :returns: Info to display
        :rtype: str
        """

        str_print = "RootNode\n\tiSAX : " + str(self.iSAX_word) + "\n\tcardinalité : " + str(self.cardinality) + \
                    "\n\tcardinalité suiv : " + str(self.cardinality_next) + "\n\tnbr nœud fils : " + \
                    str(len(self.nodes))
        return str_print


class InternalNode(RootNode):
    """
    The InternalNode class creates the internal nodes having at least one direct descendant, and a single direct ascendant

    :param tree_iSAX tree: the tree in which the node is contained
    :param Node parent: The parent parent node
    :param list sax: SAX values of the node
    :param numpy.array cardinality: Cardinality of Sax Values
    :param numpy.ndarray sequences: The sequences to be inserted in this node
    """

    def __init__(self, tree, parent, sax, cardinality, sequences):
        """
        Initialization function of the InternalNode class

        :returns: a root node
        :rtype: RootNode
        """

        """ inherits the init function of the rootnode class """
        RootNode.__init__(self, tree=tree, parent=parent,
                          sax=sax, cardinality=cardinality)

        """ transforms the list sequences from PAA"""
        list_ts_paa = self.tree.isax.transform_paa(sequences)
        tmp_mean = np_mean(list_ts_paa, axis=0)
        tmp_stdev = np_std(list_ts_paa, axis=0)

        """ as it is an internal node, it necessarily has at least one downhill node so : """
        """ we calculate the future candidate cardinalities """
        cardinality_next_tmp = np_copy(self.cardinality)
        # if max_card
        if self.tree.boolean_card_max:
            # we multiply by 2 only the cardinalities not exceeding the authorized threshold
            cardinality_next_tmp[cardinality_next_tmp <= self.tree.max_card_alphabet] *= 2
        else:
            # We multiply by 2 all the cardinalities  (they are all candidates)
            cardinality_next_tmp *= 2
        # The self.split function choses the cardinality index to multiply by 2
        position_min = self.split(cardinality_next_tmp, tmp_mean, tmp_stdev)

        """ We write the next cardinality (for its leaf nodes) """
        self.cardinality_next = np_copy(self.cardinality)
        self.cardinality_next[position_min] *= 2
        if self.tree.bigger_current_cardinality < self.cardinality_next[position_min]:
            self.tree.bigger_current_cardinality = self.cardinality_next[position_min]

        self.level = parent.level + 1

    def split(self, next_cardinality, mean, stdev):
        """
        Calcule the next cardinality and split in two

        :param numpy.array next_cardinality: The list of next cardinalities
        :param numpy.array mean: The list of averages of distribution of sequence values on each dimension
        :param numpy.array stdev: The list of different types of distribution of sequence values on each dimension
        """

        # segment_to_split : idem notation iSAX 2.0 (A Camerra, T Palpanas, J Shieh et al. - 2010)
        segment_to_split = None
        seg_to_spli_dist = float('inf')

        """ We travel the bkpts obtained for each dim and we seek the dimension that best separates our sequences
        according to our criteria """
        # List of BKPTs for chosen cardinality (ie the smallest, cf init function of InternalNode)
        bkpt_list = [self.tree.isax._card_to_bkpt_only(next_c) for next_c in next_cardinality]
        for i in range(self.tree.size_word):
            # test if we do not exceed the card max
            if next_cardinality[i] <= self.tree.max_card_alphabet and self.tree.boolean_card_max:
                # here Breakpoint closest to average of the values of the i^eth dimension
                nearest_bkpt = min(bkpt_list[i], key=lambda x: abs(x-mean[i]))
                """ 1st criterion: if the standard deviation of the values of the dimension is not zero"""
                if stdev[i] != 0:
                    """ 2nd criterion: to be the best candidate, the distance between bkpt and barycentre 
                    is divided by the gap-type """
                    if (abs((nearest_bkpt - mean[i]) / stdev[i])) < seg_to_spli_dist:
                        segment_to_split = i
                        seg_to_spli_dist = abs((nearest_bkpt - mean[i]) / stdev[i])

        """ Attention, if no candidate, choose the smallest cardinality """
        if segment_to_split is None:
            segment_to_split = np_argmin(self.cardinality)
        return segment_to_split

    def __str__(self) -> str:
        """
        Setting the display function for the node

        :returns: Info to display
        :rtype: str
        """

        str_print = "InternalNode\n\tiSAX : " + str(self.name) + "\n\tparent iSAX : " + str(self.parent.name) + \
                    "\n\tcardinality : " + str(self.cardinality) + "\n\tbext cardinality : " + \
                    str(self.cardinality_next) + "\n\tnbr node son: " + str(len(self.nodes))
        return str_print


class TerminalNode(RootNode):
    """
    The TerminalNode class creates the leaves nodes having no descendant, and a single direct ascendant

    :param tree_iSAX tree: the tree in which the node is contained
    :param Node parent: The parent parent node
    :param list sax: SAX values of the node
    :param numpy.array cardinality: Cardinality of Sax Values
    """

    def __init__(self, tree, parent, sax, cardinality):
        """
        Initialization function of the terminalnode class

        :returns: a root node
        :rtype: RootNode
        """

        RootNode.__init__(self, tree=tree, parent=parent,
                          sax=sax, cardinality=cardinality)

        del self.cardinality_next

        """ Specific part of terminal nodes
        (What? We say terminal nodes?) """
        # Variable for BKPT (non-incremental)
        self.bkpt_min, self.bkpt_max = np_array([]), np_array([])

        self.terminal = True
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

        self.splitable = True
        if np_all(np_array(self.cardinality) >= self.tree.max_card_alphabet) and self.tree.boolean_card_max:
            self.splitable = False

        """ Important, the list of PAA sequences that the tree contains"""
        self.sequences = []
        self.annotations = defaultdict(list)


    def insert_paa(self, ts_paa, annotation=None):
        """
        Function that inserts a new sequence in PAA format

        :param ts_paa: The new Paa sequence
        :param dictionary annotation: Dictionary of lists with annotations to be added to each item in a leaf       
        """

        self.sequences.append(ts_paa)
        if annotation is not None:
            for key, value in annotation.items():
                self.annotations[key].append(value)

        """ indicator maj """
        self.nb_sequences += 1
        # calculate mean and std
        if self.nb_sequences == 1:
            self.sum = np_copy(ts_paa)
            self.mean = np_copy(ts_paa)
            self.std = np_zeros(self.tree.size_word)
            self.sn = np_zeros(self.tree.size_word)
        else:
            mean_moins_1 = np_copy(self.mean)
            self.sum += ts_paa
            self.mean = self.sum / self.nb_sequences
            self.sn += (ts_paa - mean_moins_1) * (ts_paa - self.mean)
            self.std = np_sqrt(self.sn / self.nb_sequences)

    def get_annotations(self):
        """
        Returns the annotations contained in the node

        :returns: The annotations contained in the node
        :rtype: list
        """
        annotations = self.annotations
        # annotations['Node Name'] = self.short_name
        return annotations

    def get_sequences(self):
        """
        Returns the sequences contained in the node

        :returns: The sequences contained in the node
        :rtype: list
        """
        return self.sequences

    def __str__(self) -> str:
        """
        Setting the display function for the node

        :returns: Info to display
        :rtype: str
        """
        str_print = "TerminalNode\n\tiSAX : " + str(self.name) + "\n\tparent iSAX : " + str(self.parent.name) + \
                    "\n\tcardinality : " + str(self.cardinality) + "\n\tcardinality suiv : " + \
                    str(self.cardinality_next) + "\n\tnbr sequences : " + str(self.nb_sequences)
        return str_print
