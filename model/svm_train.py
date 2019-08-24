import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import dask_xgboost
import matplotlib.pyplot as plt
import utils
import xgboost
from sklearn.svm import SVC
import pickle

@click.command()
@click.option('--train-path', default='/usr/share/data/make_dataset/train')
@click.option('--model-path', default='/usr/share/data/train_model/svm.model')
def train(train_path, model_path):
  x,y = utils.load_data(train_path)

  svm_model_linear = SVC(kernel='linear', C=1).fit(x, y)
  with open(model_path, 'wb') as handle:
    pickle.dump(svm_model_linear, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()
