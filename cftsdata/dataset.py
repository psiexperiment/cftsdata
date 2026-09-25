"""
Deprecated. The loaders that used to live here now sit in `cftsdata.loaders`,
alongside the `summarize_*` module that writes the files they read, and the
filename helpers are in `cftsdata.util`:

    from cftsdata import loaders
    loaders.abr.waveforms(path)

What remains is the disk-scanning machinery of `Dataset`, which is superseded
by querying the database for the files of interest and calling the loaders on
the results. It is kept only until the last callers are migrated.
"""
import json
import os
from pathlib import Path
import zipfile

import jmespath
import pandas as pd
import yaml

try:
    from psidata.dataset import load, path_scanner
except ImportError:
    # Removed from psidata. Deferred so that importing this module (and hence
    # anything that still imports `Dataset`) does not fail outright.
    load = path_scanner = None

from cftsdata.util import parse_psi_filename


class Dataset:

    def __init__(self, ephys_path=None, subpath=None, scanner=None):
        if ephys_path is None:
            ephys_path = os.environ['PSIDATA_PROC_DIR']
        self.ephys_path = Path(ephys_path)
        self.subpath = subpath
        if subpath is not None:
            self.ephys_path = self.ephys_path / subpath
        if not self.ephys_path.exists():
            raise ValueError(f'Unknown data path {self.ephys_path}')
        self.raw_ephys_path = self.ephys_path
        if scanner is None:
            if path_scanner is None:
                raise RuntimeError(
                    'psidata.dataset.path_scanner is no longer available. '
                    'Query the database for the files of interest and call '
                    'the loaders in cftsdata.loaders on the result instead.')
            scanner = path_scanner(self.ephys_path)
        self.scanner = scanner

    def load_raw(self, cb, etype=None, **kwargs):
        '''
        Facilitate loading information from raw data zipfile

        Parameters
        ----------
        cb : callable
            Takes a single argument, the zip filename, and returns a dictionary
            or dataframe.
        '''
        wildcard = '**/*.zip' if etype is None else f'**/*{etype}*.zip'
        return self.load(cb, wildcard, data_path=self.raw_ephys_path, **kwargs)

    def load_experiment_info(self, **kwargs):
        '''
        Load experiment info

        Includes `filename` as a column.
        '''
        return self.load_raw(lambda x: {'filename': x},
                             filename_parser=parse_psi_filename,
                             **kwargs)

    def load_csv_header(self, filename, index_col=0, etype=None, **kwargs):
        def cb(zip_filename):
            nonlocal filename
            with zipfile.ZipFile(zip_filename) as fh_zip:
                with fh_zip.open(filename) as fh:
                    return pd.read_csv(fh, nrows=1, index_col=index_col).iloc[0].to_dict()
        return self.load_raw(cb, etype, **kwargs)

    def load_raw_jmes(self, filename, query, etype=None, file_format=None,
                      **kwargs):
        '''
        Load value from JSON or YAML file saved to the raw data zipfile

        Parameters
        ----------
        filename : str
            Name of file stored in zipfile to query (e.g., `io.json`,
            `final.preferences`).
        query : dict
            Mapping of result names to the corresponding JMES query.
        etype : {None, str}
            If specified, filter datasets by those that match the experiment
            type (e.g., abr_io).
        file_format : {None, 'json', 'yaml'}
            File format. If None, will make a best guess based on the filename
            ending.

        Examples
        --------
        Load the channel used for the starship A primary output.
        >>> ds.load_raw_jmes('io.json', {'primary_output': 'output.starship_A_primary.channel'})
        '''
        c_query = {n: jmespath.compile(q) for n, q in query.items()}
        if file_format is None:
            if filename.endswith('.json'):
                file_format = 'json'
            elif filename.endswith('.yaml'):
                file_format = 'yaml'
            elif filename.endswith('.preferences'):
                file_format = 'yaml'
            else:
                raise ValueError(f'Could not determine file format for {filename}')
        if file_format not in ('json', 'yaml'):
            raise ValueError(f'Invalid file format {file_format}')

        def cb(zip_filename):
            nonlocal c_query
            nonlocal filename
            nonlocal file_format
            with zipfile.ZipFile(zip_filename) as fh_zip:
                with fh_zip.open(filename) as fh:
                    if file_format == 'json':
                        text = json.load(fh)
                    elif file_format == 'yaml':
                        text = yaml.safe_load(fh)
                    return {n: q.search(text) for n, q in c_query.items()}
        return self.load_raw(cb, etype, **kwargs)

    def load(self, cb, glob, filename_parser=None, data_path=None,
             include_dataset=False, should_load_cb=None, info_as_cols=True):
        '''
        Parameters
        ----------
        cb : callable
            Callback that takes name of file and returns a DataFrame or Series.
        glob : string
            Wildcard pattern used to find files to load.
        filename_parser : {None, callable}
            Callback that returns dictionary containing keys that will be added
            as columns to the result.
        data_path : {None, string, Path}
            Path to scan for data. If not provided, defaults to the ephys path.
        include_dataset : bool
            If True, include the name of the dataset the file was found in
            (i.e., the parent folder).
        should_load_cb : {None, callable}
            Callback that returns True if the file should be loaded. If a
            callback is not provided, all files found are loaded.
        '''
        if data_path is None:
            data_path = self.ephys_path
        if filename_parser is None:
            filename_parser = parse_psi_filename
        scanner = self.scanner(glob)
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('glob')
        kwargs.pop('data_path')
        return load(**kwargs)

    def load_iec_psd(self, **kwargs):
        etype = 'inear_speaker_calibration_chirp'
        result = self.load(
            lambda x: pd.read_csv(x),
            '**/*inear_speaker_calibration_chirp psd.csv',
            parse_psi_filename,
            **kwargs
        )
        query = {
            'starship_A': 'input.starship_A_microphone.calibration.attrs.name',
            'starship_B': 'input.starship_B_microphone.calibration.attrs.name',
        }
        starship = self.load_raw_jmes('io.json', query, etype, **kwargs) \
            .set_index('datetime', verify_integrity=True)[['starship_A', 'starship_B']]
        metadata = self.load_csv_header('epoch_metadata.csv', etype=etype) \
            .set_index('datetime', verify_integrity=True)[['system', 'system_output']]
        merged = result.join(starship, on='datetime').join(metadata, on='datetime')
        merged['starship'] = merged.apply(lambda x: x[x['system']], axis=1)
        return merged.drop(['system', 'starship_A', 'starship_B'], axis=1)

    def load_memr_system(self, etype='memr', **kwargs):
        '''
        Load the outputs used for the probe and elicitor
        '''
        query = {
            'elicitor': "values(output)[?outputs[?name == 'elicitor_secondary']].name | [0]",
            'probe': "values(output)[?outputs[?name == 'probe_primary']].name | [0]",
        }
        return self.load_raw_jmes('io.json', query, etype=etype)

    def _get_memr_etype(self, memr):
        memr_map = {
            'valero': 'memr_simultaneous_chirp',
            'keefe': 'memr_interleaved_click',
            'sweep': 'memr_sweep_click',
        }
        return memr_map[memr.lower()]

    def load_memr_level(self, memr):
        etype = self._get_memr_etype(memr)
        glob = f'**/*{etype} raw_levels.csv'
        return self.load(lambda x: pd.read_csv(x), glob, parse_psi_filename)
