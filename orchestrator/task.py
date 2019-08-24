import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def result_path(self):
        return str(f'{self.out_dir}/{self.fname}.csv')

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=self.result_path()
        )


class MakeDatasets(DockerTask):
    out_dir = luigi.Parameter(default='/usr/share/data/make_dataset/')
    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        return [
            'python', 'dataset.py',
            '--in-csv', DownloadData().result_path(),
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=str(out_dir / '.SUCCESS')
        )

class TrainModel(DockerTask):
    train_path = luigi.Parameter(default='/usr/share/data/make_dataset/train')
    out_dir = luigi.Parameter(default='/usr/share/data/train_model/')

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    def result_path(self):
        return str(f'{self.out_dir}/xgb.model')

    @property
    def command(self):
        return [
            'python', 'train.py',
            '--train-path', self.train_path,
            '--model-path', self.result_path()
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=self.result_path()
        )

class TrainSvmModel(DockerTask):
    train_path = luigi.Parameter(default='/usr/share/data/make_dataset/train')
    out_dir = luigi.Parameter(default='/usr/share/data/train_model/')

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    def result_path(self):
        return str(f'{self.out_dir}svm.model')

    @property
    def command(self):
        return [
            'python', 'svm_train.py',
            '--train-path', self.train_path,
            '--model-path', self.result_path()
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=self.result_path()
        )


class EvaluateModel(DockerTask):
    eval_path = luigi.Parameter(default='/usr/share/data/make_dataset/test')
    out_dir = luigi.Parameter(default='/usr/share/data/eval_model/')

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return [
            'python', 'eval.py',
            '--eval-path', self.eval_path,
            '--model-path', TrainModel().result_path(),
            '--out-dir', self.out_dir,
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=str(out_dir / '.SUCCESS')
        )

class EvaluateSvmModel(DockerTask):
    eval_path = luigi.Parameter(default='/usr/share/data/make_dataset/test')
    out_dir = luigi.Parameter(default='/usr/share/data/eval_svm_model/')

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return TrainSvmModel()

    @property
    def command(self):
        return [
            'python', 'svm_eval.py',
            '--eval-path', self.eval_path,
            '--model-path', TrainSvmModel().result_path(),
            '--out-dir', self.out_dir,
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=str(out_dir / '.SUCCESS')
        )

class Report(DockerTask):
    out_dir = luigi.Parameter(default='/usr/share/data/report/')

    @property
    def image(self):
        return f'code-challenge/model:{VERSION}'

    def requires(self):
        return [EvaluateSvmModel(),EvaluateModel()]

    def result_path(self):
        return str(f'{self.out_dir}/.SUCCESS')

    @property
    def command(self):
        return [
            'python', 'report.py',
            '--report-path', self.out_dir,
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(
            path=self.result_path()
        )
