import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import dask_xgboost
import matplotlib.pyplot as plt
import utils
import xgboost

def preprocess(df):
  train_y = df['group'].values
  df = df.drop('group', axis=1)

  df['country'] = df['country'].astype('category')
  df = df.categorize()
  enc = DummyEncoder(['country'])
  enc.fit(df)
  train_x = enc.transform(df)
  return train_x, train_y

@click.command()
@click.option('--train-path', default='/usr/share/data/make_dataset/train.parquet')
@click.option('--model-path', default='/usr/share/data/train_model/xgb.model')
def train(train_path, model_path):
  client = Client('dask-scheduler:8786')
  params = {'objective': 'multi:softmax', 'num_class': 4,
            'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
            'min_child_weight': 0.5}

  df = dd.read_parquet(train_path)
  train_x, train_y = utils.preprocess(df)
  bst = dask_xgboost.train(client, params, train_x, train_y, num_boost_round=10)
  bst.save_model(model_path)

if __name__ == '__main__':
    train()
