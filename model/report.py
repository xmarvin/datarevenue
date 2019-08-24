import click
from pathlib import Path

@click.command()
@click.option('--report-path', default='/usr/share/data/report/')
def report(report_path):
  out_dir = Path(report_path)
  flag = out_dir / '.SUCCESS'
  flag.touch()

if __name__ == '__main__':
    report()