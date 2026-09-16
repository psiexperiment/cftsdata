import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import psidata.version
import cftsdata.version

from . import loaders
from .util import parse_psi_filename


def _find_abr_datasets(path=None, subpath=None):
    '''
    Find the processed ABR datasets under `path`

    Parameters
    ----------
    path : {None, str, pathlib.Path}
        Root to scan. Defaults to the `PROC_DATA_DIR` environment variable.
    subpath : {None, str}
        If provided, scan only this subfolder of `path`.

    Returns
    -------
    path : pathlib.Path
        The root that was scanned. Zip entries are stored relative to it.
    folders : list of pathlib.Path
        Dataset folders containing ABR average waveforms, sorted by name.

    Notes
    -----
    `summarize_mlr_llr` writes a file of the same name in a different layout,
    so datasets are filtered by experiment type rather than by filename alone.
    '''
    if path is None:
        path = os.environ['PROC_DATA_DIR']
    path = Path(path)
    if subpath is not None:
        path = path / subpath
    if not path.exists():
        raise ValueError(f'Unknown data path {path}')

    folders = set()
    for filename in path.rglob(f'*{loaders.abr.SUFFIXES["waveforms"]}'):
        folder = filename.parent
        try:
            info = parse_psi_filename(folder)
        except ValueError:
            continue
        if info['experiment_type'].startswith('abr'):
            folders.add(folder)
    return path, sorted(folders)


def zip_unrated_abr_data():
    '''
    Create a zipfile of processed ABR data that needs to be reviewed

    The generated zipfile only includes the subset of data needed to
    successfully run the ABR peak-picking program.
    '''
    import argparse
    parser = argparse.ArgumentParser('cfts-zip-unrated-abr-data')
    parser.add_argument('-p', '--path', type=Path)
    parser.add_argument('-s', '--subpath', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-x', '--exclude', type=str, nargs='*', default=[])
    parser.add_argument('-d', '--min-date', type=str)
    parser.add_argument('raters', type=str, nargs='+')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    # Expand the exclude argument
    exclude = []
    for path in args.exclude:
        path = Path(path)
        if '*' in path.name:
            exclude.extend(path.parent.glob(path.name))
        else:
            exclude.append(path)

    def _get_id(dataset_path):
        return Path(dataset_path).stem.split(' ')[0]

    # Make a list of folders already in an existing zip archive so that we can
    # avoid adding them to the new zip archive.
    exclude_ids = set()
    for filename in exclude:
        for name in zipfile.ZipFile(filename).namelist():
            if name.endswith('average waveforms.csv'):
                exclude_ids.add(_get_id(name))

    ephys_path, folders = _find_abr_datasets(args.path, args.subpath)

    dataset_map = {_get_id(f): f for f in folders}
    all_datasets = set((_get_id(f), frequency) for f in folders
                       for frequency in loaders.abr.frequencies(f))

    if args.min_date is not None:
        min_date = dt.datetime.strptime(args.min_date, '%Y%m%d')
        all_datasets = set((d, f) for d, f in all_datasets if dt.datetime.strptime(d[:8], '%Y%m%d') > min_date)

    # Filter out datasets that are already rated by one of the raters
    rated_datasets = set()
    for folder in folders:
        th = loaders.abr.th(folder)
        for _, row in th.iterrows():
            if row['rater'] in args.raters:
                rated_datasets.add((_get_id(folder), row['frequency']))

    unrated_ids = set(ds[0] for ds in (all_datasets - rated_datasets))
    include_ids = unrated_ids - exclude_ids
    include_folders = [dataset_map[n] for n in include_ids]

    if args.verbose:
        print(f'Found {len(unrated_ids)} experiments to process. '
              f'Excluding {len(exclude_ids)} from final zip folder.')
        for folder in sorted(include_folders):
            print(f' â€¢ {folder}')

    if len(include_folders):
        _zip_abr_folders(include_folders, args.output, ephys_path, args.raters)


def _zip_abr_folders(folders, output, relative_path, raters=None):
    if raters is None:
        raters = ['']
    glob_patterns = [
        '*average waveforms.csv',
        '*ABR processing settings.json',
    ] + [f'*{r}-analyzed.txt' for r in raters]
    with zipfile.ZipFile(output, 'w') as fh:
        for folder in folders:
            for pattern in glob_patterns:
                for filename in folder.glob(pattern):
                    fh.write(filename, str(filename.relative_to(relative_path)))


def zip_multirater_abr_data():
    '''
    Create a zipfile of processed ABR data scored by more than one rater.

    The generated zipfile is used for comparing raters.
    '''
    import argparse
    parser = argparse.ArgumentParser('cfts-zip-multirater-abr-data')
    parser.add_argument('-p', '--path', type=Path)
    parser.add_argument('-s', '--subpath', type=str)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    ephys_path, folders = _find_abr_datasets(args.path, args.subpath)

    datasets = []
    for folder in folders:
        th = loaders.abr.th(folder)
        if th.empty:
            continue
        if th.groupby('frequency')['rater'].nunique().gt(1).any():
            datasets.append(folder)
    _zip_abr_folders(datasets, args.output, ephys_path)


def invert_eeg_data():
    '''
    Fix EEG data (e.g., from EFR and ABR experiments) where the pinna and
    vertex electrodes were switched.
    '''
    import argparse
    import zarr
    parser = argparse.ArgumentParser('cfts-invert-eeg-data')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempdir:
        archive_path = Path(tempdir) / args.path.stem
        if args.path.suffix != '.zip':
            # TOOD: Add support for this when needed
            raise ValueError('Currently this only works with zip files')
        else:
            with zipfile.ZipFile(args.path, 'r') as zip_fh:
                zip_fh.extractall(archive_path)
                zarr_path = archive_path / 'eeg.zarr'
                old_zarr_path = archive_path / 'eeg-backup.zarr'
                zarr_path.rename(old_zarr_path)

                old_array = zarr.open(old_zarr_path)
                new_array = zarr.array(-old_array[:], store=zarr_path)
                for k, v in old_array.attrs.items():
                    new_array.attrs[k] = v
                flag_file = archive_path / 'eeg-corrections.json'
                detail = {
                    'note': 'Corrected for inverted polarity by cftsdata.postprocess.invert_eeg_data',
                    'versions': {
                        'cftsdata': cftsdata.version.__version__,
                        'psidata': psidata.version.__version__,
                    }
                }
                flag_file.write_text(json.dumps(detail, indent=4))
                archive_data(archive_path)
                zip_archive_path = archive_path.with_suffix('.zip')
                shutil.copyfile(zip_archive_path, args.path)
