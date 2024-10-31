import datetime as dt
import hashlib
import json
import shutil
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from tqdm import tqdm

import psidata.version
import cftsdata.version

from .dataset import Dataset


def archive_data(path):
    '''
    Creates a zip archive of the specified path, validates the MD5sum of each
    file in the archive, and then generates an MD5sum sidecar containing the
    MD5sum of the zip file itself.
    '''
    # Need to handle cases where path has a dot in it.
    zippath = Path(str(path) + '.zip')
    md5path = Path(str(path) + '.md5')
    path = Path(path)
    shutil.make_archive(str(path), 'zip', str(path))

    try:
        validate(zippath)
        zipmd5 = md5sum(zippath.open('rb'))
        md5path.write_text(zipmd5)
        shutil.rmtree(path)
    except IOError as e:
        print(e)


def zip_data():
    '''
    Main function for creating zipfiles from raw CFTS data

    For each CFTS experiment, compress into a single zipfile that is supported
    by `psidata.Recording`. Calculate MD5sum of zipfile and save alongside the
    zipfile. The MD5sum can be safely discarded once the zipfile has been
    transferred to a filesystem that uses checksums for integrity checks such
    as BTRFS or ZFS.
    '''
    import argparse
    parser = argparse.ArgumentParser('cfts-zip-data')
    parser.add_argument('path', type=Path)
    parser.add_argument('-d', '--destination', type=Path)
    args = parser.parse_args()

    # Make zip archives first
    dirs = [p for p in args.path.iterdir() if p.is_dir()]
    for path in tqdm(dirs):
        archive_data(path)

    # Now, move all zip and md5 files if a destination is specified
    if args.destination is not None:
        for zippath in tqdm(args.path.glob('*.zip')):
            md5path = zippath.with_suffix('.md5')
            for file in (zippath, md5path):
                new_file = args.destination / file.name
                file.rename(new_file)


def md5sum(stream, blocksize=1024**2):
    '''
    Generates md5sum from byte stream

    Parameters
    ----------
    stream : stream
        Any object supporting a `read` method that returns bytes.
    blocksize : int
        Blocksize to use for computing md5sum

    Returns
    -------
    md5sum : str
        Hexdigest of md5sum for stream
    '''
    md5 = hashlib.md5()
    while True:
        block = stream.read(blocksize)
        if not block:
            break
        md5.update(block)
    return md5.hexdigest()


def validate(path):
    '''
    Validates contents of zipfile using md5sum

    Parameters
    ----------
    path : {str, pathlib.Path}
        Path containing data that was zipped. Zipfile is expected to have the
        same path, but ending in ".zip".

    The zipfile is opened and iterated through. The MD5 sum for each file
    inside the archive is compared with the companion file in the unzipped
    folder.
    '''
    archive = zipfile.ZipFile(path)
    for name in archive.namelist():
        archive_md5 = md5sum(archive.open(name))
        file = path / name
        if file.is_file():
            with file.open('rb') as fh:
                file_md5 = md5sum(fh)
            if archive_md5 != file_md5:
                raise IOError('{name} in zipfile for {path} is corrupted')


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

    dataset = Dataset(ephys_path=args.path, subpath=args.subpath)

    freqs = dataset.load_abr_frequencies(include_dataset=True)
    dataset_map = {_get_id(r['dataset']): r['dataset'] for _, r in freqs.iterrows()}
    all_datasets = set((_get_id(r['dataset']), r['frequency']) for _, r in freqs.iterrows())

    if args.min_date is not None:
        min_date = dt.datetime.strptime(args.min_date, '%Y%m%d')
        all_datasets = set((d, f) for d, f in all_datasets if dt.datetime.strptime(d[:8], '%Y%m%d') > min_date)

    # Filter out datasets that are already rated by one of the raters
    rated_datasets = set()
    for rater in args.raters:
        th = dataset.load_abr_th(rater, include_dataset=True)
        rated_datasets |= set((_get_id(r['dataset']), r['frequency']) for _, r in th.iterrows())

    unrated_ids = set(ds[0] for ds in (all_datasets - rated_datasets))
    include_ids = unrated_ids - exclude_ids
    include_folders = [dataset_map[n] for n in include_ids]

    if args.verbose:
        print(f'Found {len(unrated_ids)} experiments to process. '
              f'Excluding {len(exclude_ids)} from final zip folder.')
        for folder in sorted(include_folders):
            print(f' • {folder}')

    if len(include_folders):
        _zip_abr_folders(include_folders, args.output, dataset.ephys_path, args.raters)


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
    dataset = Dataset(ephys_path=args.path, subpath=args.subpath)

    th = dataset.load_abr_th(include_dataset=True)
    n_raters = th.groupby(['dataset', 'frequency'])['rater'].nunique()
    multiple_raters = n_raters[n_raters > 1]
    datasets = multiple_raters.index.unique('dataset')
    _zip_abr_folders(datasets, args.output, dataset.ephys_path)


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


def merge_pdf():
    '''
    Merge PDFs matching filename pattern into a single file for review
    '''
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import FreeText
    from pypdf.errors import PdfReadError
    import argparse

    parser = argparse.ArgumentParser('cfts-merge-pdf')
    parser.add_argument('path', type=Path)
    parser.add_argument('pattern', type=str)
    parser.add_argument('output', type=Path)
    parser.add_argument('--delete-bad', action='store_true')
    args = parser.parse_args()

    if args.output.exists():
        raise IOError('Output file exists')

    # Make sure we can create the file before we get too far along.
    args.output.touch()
    args.output.unlink()

    pattern = args.pattern
    if not pattern.endswith('.pdf'):
        pattern = pattern + '.pdf'
    if not pattern.startswith('**/'):
        pattern = '**/' + pattern

    merger = PdfWriter()
    for filename in tqdm(args.path.glob(pattern)):
        if '_exclude' in str(filename):
            # Don't process files hidden under the _exclude folders.
            continue
        try:
            fh = PdfReader(filename)
            merger.append(fh, filename.stem)
        except PdfReadError:
            if args.delete_bad:
                filename.unlink()
            print(filename)
    merger.write(args.output)
