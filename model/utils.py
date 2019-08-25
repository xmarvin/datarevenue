import numpy as np
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import sparse
from pathlib import Path

def load_data(path):
  dx = scipy.sparse.load_npz(path + '_x.npz')
  dy = np.load(path + '_y.npy')
  return dx,dy


def save_output(pred_y, y, out_dir: Path):
    matrix = confusion_matrix(y, pred_y)
    acc = accuracy_score(y, pred_y)
    f1 = f1_score(y, pred_y, average='weighted')
    matrix_path = out_dir / 'confusion_matrix.txt'
    with open(matrix_path, 'w') as f:
      print(matrix, file=f)

      metrics_path = out_dir / 'metrics.txt'
      with open(metrics_path, 'w') as f:
        print('Accuracy: {0:.2f}'.format(acc), file=f)
        print('F1: {0:.2f}'.format(f1), file=f)

        flag = out_dir / '.SUCCESS'
        flag.touch()