import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import dask_xgboost
import matplotlib.pyplot as plt
import utils
import xgboost
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def save_output(pred_y, y, out_dir: Path):
    matrix = confusion_matrix(y.compute(), pred_y.astype(int).compute())
    acc = accuracy_score(y.compute(), pred_y.astype(int).compute())
    f1 = f1_score(y.compute(), pred_y.astype(int).compute(), average='weighted')
    print(matrix)
    print(acc)
    #TODO: Add picture version
    matrix_path = out_dir / 'confusion_matrix.txt'
    with open(matrix_path, 'w') as f:
      print(matrix, file=f)

    metrics_path = out_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
      print('Accuracy: {0:.2f}'.format(acc), file=f)
      print('F1: {0:.2f}'.format(f1), file=f)

    flag = out_dir / '.SUCCESS'
    flag.touch()

@click.command()
@click.option('--eval-path', default='/usr/share/data/make_dataset/test.parquet')
@click.option('--model-path', default='/usr/share/data/train_model/xgb.model')
@click.option('--out-dir', default='/usr/share/data/eval_model/')
def eval(eval_path, model_path, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  client = Client('dask-scheduler:8786')
  df = dd.read_parquet(eval_path)
  x, y = utils.preprocess(df)
  bst = xgboost.Booster()
  bst.load_model(model_path)
  pred_y =  dask_xgboost.predict(client, bst, x)
  save_output(pred_y, y, out_dir)

if __name__ == '__main__':
    eval()
