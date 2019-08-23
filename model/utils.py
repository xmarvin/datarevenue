import dask.dataframe as dd
import numpy as np
from dask_ml.preprocessing import DummyEncoder

def preprocess(df):
  train_y = df['group'].values
  df = df.drop('group', axis=1)

  df['country'] = df['country'].astype('category')
  df = df.categorize()
  enc = DummyEncoder(['country'])
  enc.fit(df)
  train_x = enc.transform(df)
  return train_x, train_y