import click
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import utils
from sklearn.linear_model import LogisticRegression
import pickle

@click.command()
@click.option('--train-path', default='/usr/share/data/make_dataset/train')
@click.option('--model-path', default='/usr/share/data/train_model/lr.model')
def train(train_path, model_path):
  x,y = utils.load_data(train_path)
  lr = LogisticRegression()
  lr.fit(x, y)

  with open(model_path, 'wb') as handle:
    pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()
