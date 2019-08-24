import numpy as np
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import dask.dataframe as dd
import dask.array as da
import sparse
from pathlib import Path

def load_data(path):
  train_x = scipy.sparse.load_npz(path + '_x.npz')
  train_y = np.load(path + '_y.npy')
  # dx = da.from_array(train_x, chunks=1000)
  # dx = dx.map_blocks(sparse.COO)
  # TODO: Find a way to read sparse into dask
  dx = da.from_array(train_x.todense(), chunks=10000)
  dy = da.from_array(train_y, chunks=10000)
  return dx,dy


def save_output(pred_y, y, out_dir: Path):
    matrix = confusion_matrix(y, pred_y)
    acc = accuracy_score(y, pred_y)
    f1 = f1_score(y, pred_y, average='weighted')
    # TODO: Add picture version
    matrix_path = out_dir / 'confusion_matrix.txt'
    with open(matrix_path, 'w') as f:
      print(matrix, file=f)

      metrics_path = out_dir / 'metrics.txt'
      with open(metrics_path, 'w') as f:
        print('Accuracy: {0:.2f}'.format(acc), file=f)
        print('F1: {0:.2f}'.format(f1), file=f)

        flag = out_dir / '.SUCCESS'
        flag.touch()