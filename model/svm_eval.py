import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import dask_xgboost
import matplotlib.pyplot as plt
import utils
import pickle

@click.command()
@click.option('--eval-path', default='/usr/share/data/make_dataset/test')
@click.option('--model-path', default='/usr/share/data/train_model/svm.model')
@click.option('--out-dir', default='/usr/share/data/eval_svm_model/')
def eval(eval_path, model_path, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  x,y = utils.load_data(eval_path)

  with open(model_path, 'rb') as handle:
    svm_model_linear = pickle.load(handle)
  pred_y = svm_model_linear.predict(x)

  utils.save_output(pred_y, y, out_dir)

if __name__ == '__main__':
    eval()
