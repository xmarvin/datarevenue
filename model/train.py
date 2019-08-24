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
@click.option('--train-path', default='/usr/share/data/make_dataset/train')
@click.option('--model-path', default='/usr/share/data/train_model/xgb.model')
def train(train_path, model_path):
  client = Client('dask-scheduler:8786')
  params = {'objective': 'multi:softmax', 'num_class': 4,
            'max_depth': 12, 'eta': 0.01, 'subsample': 0.7,
            'min_child_weight': 0.5}

  x,y = utils.load_data(train_path)
  bst = dask_xgboost.train(client, params, x, y, num_boost_round=500)
  bst.save_model(model_path)

if __name__ == '__main__':
    train()
