import click
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import utils
import pickle
from sklearn.linear_model import LogisticRegression

@click.command()
@click.option('--eval-path', default='/usr/share/data/make_dataset/test')
@click.option('--model-path', default='/usr/share/data/train_model/lr.model')
@click.option('--out-dir', default='/usr/share/data/eval_model/')
def eval(eval_path, model_path, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  x,y = utils.load_data(eval_path)

  with open(model_path, 'rb') as handle:
    lr = pickle.load(handle)
  pred_y = lr.predict(x)

  utils.save_output(pred_y, y, out_dir)

if __name__ == '__main__':
    eval()
