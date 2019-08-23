import click
import numpy as np
import dask.dataframe as dd
from dask_ml.preprocessing import OneHotEncoder
from distributed import Client
from pathlib import Path

def _save_dataset(df, name, outdir: Path):
    out_path = outdir / name
    df.to_parquet(str(out_path))

def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""

    _save_dataset(train, 'train.parquet/', outdir)
    _save_dataset(test, 'test.parquet/', outdir)

    flag = outdir / '.SUCCESS'
    flag.touch()

def make_features(df):
    df['description_len'] = df['description'].apply(len)
    #useful_columns = ['price', 'country', 'description', 'description_len', 'group']
    useful_columns = ['price', 'country', 'description_len', 'group']
    df = df[useful_columns]
    df['price'] = df['price'].fillna(0)
    return df

def make_groups(df):
    def points2group(score):
        splits = [85, 90, 95]
        for (index, split) in enumerate(splits):
            if score < split:
                return index
        return len(splits)

    df['group'] = df['points'].apply(points2group)
    return df

@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--train-ratio', default=0.8, type=float)
def make_datasets(in_csv, out_dir, train_ratio):
    test_ratio = 1 - train_ratio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect to the dask cluster
    c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    ddf = dd.read_csv(in_csv, blocksize=1e6)
    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')
    ddf = make_groups(ddf)
    ddf = make_features(ddf)
    train, test = ddf.random_split([train_ratio, test_ratio], random_state=0)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
