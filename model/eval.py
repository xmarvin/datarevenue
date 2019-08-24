import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import dask_xgboost
import matplotlib.pyplot as plt
import utils
import xgboost

@click.command()
@click.option('--eval-path', default='/usr/share/data/make_dataset/test')
@click.option('--model-path', default='/usr/share/data/train_model/xgb.model')
@click.option('--out-dir', default='/usr/share/data/eval_model/')
def eval(eval_path, model_path, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  client = Client('dask-scheduler:8786')
  x,y = utils.load_data(eval_path)
  bst = xgboost.Booster()
  bst.load_model(model_path)
  pred_y = dask_xgboost.predict(client, bst, x)
  pred_y = pred_y.astype(int).compute()
  y = y.compute()
  utils.save_output(pred_y, y, out_dir)

if __name__ == '__main__':
    eval()
