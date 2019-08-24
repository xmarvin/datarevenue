import click
import numpy as np
import dask.dataframe as dd
from distributed import Client
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

def _save_dataset(x, y, name, outdir: Path):
    scipy.sparse.save_npz(str(outdir / (name + '_x')), x)
    np.save(outdir / (name+'_y'), y)

def _save_datasets(all_data, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""

    x_train, x_test, y_train, y_test = all_data
    _save_dataset(x_train, y_train,'train', outdir)
    _save_dataset(x_test, y_test, 'test', outdir)

    flag = outdir / '.SUCCESS'
    flag.touch()

def make_features(df):
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
    stop_words = text.ENGLISH_STOP_WORDS
    def tokenize(text):
        return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower()) if word not in stop_words]

    df['country'] = df['country'].fillna('unk')
    df['price'] = df['price'].fillna(df['price'].mean().compute())

    df['country'] = df['country'].astype('category')
    df = df.categorize(columns=['country'])
    df = df.compute()

    vectorizer = TfidfVectorizer(tokenizer = tokenize, max_features = 1000)
    descriptions = df['description'].values
    vectorizer.fit(descriptions)
    description_encodings = vectorizer.transform(descriptions)
    country_encodings = csr_matrix(dd.get_dummies(df['country'], prefix='country'))
    other_columns = csr_matrix(df[['price']])
    all_data = hstack((description_encodings, country_encodings, other_columns))
    return all_data

def make_groups(df):
    def points2group(score):
        splits = [84, 89, 95]
        for (index, split) in enumerate(splits):
            if score < split:
                return index
        return len(splits)

    return df['points'].apply(points2group, meta=('points','int')).values.compute()

@click.command()
@click.option('--in-csv', default='/usr/share/data/raw/wine_dataset')
@click.option('--out-dir', default = '/usr/share/data/make_dateset/')
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
    groups = make_groups(ddf)
    data = make_features(ddf)
    all_data = train_test_split(data, groups, test_size=test_ratio,random_state=0)
    _save_datasets(all_data, out_dir)


if __name__ == '__main__':
    make_datasets()
