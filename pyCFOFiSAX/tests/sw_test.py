import cdflib
import numpy as np

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from multiprocessing import Pool
from scipy.stats import norm

from astropy.time import Time 

import spiceypy as spice
from mpl_toolkits.mplot3d import Axes3D 


from pyts.approximation import SymbolicAggregateApproximation


import sys
sys.path.insert(0, '/home/andres_munoz_j/pyCFOFiSAX')

print(sys.path)

# import importlib.util
# spec = importlib.util.spec_from_file_location('ForestISAX', '/home/andres_munoz_j/pyCFOFiSAX/pyCFOFiSAX/_forest_iSAX.py')
# ForestISAX = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(ForestISAX)

# spec = importlib.util.spec_from_file_location('TreeISAX', '/home/andres_munoz_j/pyCFOFiSAX/pyCFOFiSAX/_tree_iSAX.py')
# TreeISAX = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(TreeISAX)

from pyCFOFiSAX._forest_iSAX import ForestISAX
from pyCFOFiSAX._isax import IndexableSymbolicAggregateApproximation
# from pyCFOFiSAX._tree_iSAX import TreeISAX
from anytree import RenderTree

from anytree.exporter import DotExporter

psp_path = '/sw-data/psp/'
path = psp_path + 'mag_rtn/'
year = '2019'
month = '05'
day = '15'
hour = '00'

cdf_file_path = path + year + '/psp_fld_l2_mag_rtn_' + year + month + day + hour + '_v01.cdf'

cdf_file = cdflib.CDF(cdf_file_path)


x = cdf_file.varget('epoch_mag_RTN')    # reading in the epoch time stamps
x = cdflib.epochs.CDFepoch.to_datetime(x) # convrting x axis labels to date time stamps
y = cdf_file.varget('psp_fld_l2_mag_RTN')

npoints = 200

# Start with Bx
ts = y[0:int(y.shape[0]/npoints)*npoints,0].reshape(-1,npoints)
# Append By
ts = np.append(ts, y[0:int(y.shape[0]/npoints)*npoints,1].reshape(-1,npoints), axis=0)
# Append Bz
ts = np.append(ts, y[0:int(y.shape[0]/npoints)*npoints,2].reshape(-1,npoints), axis=0)

# Create auxiliary dataframe
ts_loc = pd.DataFrame({'File':np.repeat(cdf_file_path,ts.shape[0]/3), 'Component':np.repeat('Bx',ts.shape[0]/3)})
ts_loc['t0'] = np.array(x[0:int(y.shape[0]/npoints)*npoints]).reshape(-1,npoints)[:,0]
ts_loc['t1'] = np.array(x[0:int(y.shape[0]/npoints)*npoints]).reshape(-1,npoints)[:,-1]
tmp_loc = ts_loc.copy()
tmp_loc['Component'] = 'By'
ts_loc = pd.concat((ts_loc,tmp_loc))
tmp_loc['Component'] = 'Bz'
ts_loc = pd.concat((ts_loc,tmp_loc)).reset_index(drop=True)

sw_forest = ForestISAX(size_word=10,
                           threshold=20,
                           data_ts=ts,
                           base_cardinality=2, number_tree=1)

sw_forest.index_data(ts, annotation=ts_loc)

size_word = 10
mu = np.mean(ts)
sig = np.std(ts)

isax = IndexableSymbolicAggregateApproximation(size_word, mean=mu, std=sig)

nodes_at_level = sw_forest.forest[0].get_nodes_of_level_or_terminal(8)

annotations_l = nodes_at_level[30].get_annotations()
sequences_l = nodes_at_level[30].get_sequences()
annotations_l = pd.concat([pd.Series(sequences_l, index=annotations_l.index, name='iSAX'), annotations_l], axis=1)

# print(sw_forest.forest[0].root.get_sequences())
# print(sw_forest.forest[0].root.get_annotations())

print('done') 