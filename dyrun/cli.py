import click
import os
import shutil


def build(dst_dir):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(src_dir, dst_dir, ignore=_ignore_files)

def _ignore_files(dir, files):
    ignored_files = ['utils.py', 'cli.py']
    ignored_dirs = ['__pycache__']
    return [
        f for f in files
        if f in ignored_files or os.path.isdir(os.path.join(dir, f)) and f in ignored_dirs
    ]

@click.command()
@click.argument('folder_name', default='dyrun')
def build_cmd(folder_name):
    """
    Build the dyrun package
    """
    dst_dir = os.path.join(os.getcwd(), folder_name)
    build(dst_dir)
    click.echo(f'dyrun package built successfully in {dst_dir}!')