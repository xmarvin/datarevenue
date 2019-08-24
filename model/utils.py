import numpy as np
import scipy

import dask.dataframe as dd
import dask.array as da
import sparse

def load_data(path):
  train_x = scipy.sparse.load_npz(path + '_x.npz')
  train_y = np.load(path + '_y.npy')
  # dx = da.from_array(train_x, chunks=1000)
  # dx = dx.map_blocks(sparse.COO)
  # TODO: Find a way to read sparse into dask
  dx = da.from_array(train_x.todense(), chunks=10000)
  dy = da.from_array(train_y, chunks=10000)
  return dx,dy