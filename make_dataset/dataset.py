import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--train-ratio', default=0.8, type=float)
def make_datasets(in_csv, out_dir, train_ratio):

    test_ratio = 1 - train_ratio
    print(train_ratio, test_ratio)
    return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect to the dask cluster
    c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    ddf = dd.read_csv(in_csv, blocksize=1e6)
    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')
    train, test = ddf.random_split([train_ratio, test_ratio], random_state=0)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
