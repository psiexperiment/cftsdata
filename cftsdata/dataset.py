import datetime as dt
from functools import partial
import json
import re
import os
from pathlib import Path
import zipfile

import jmespath
import pandas as pd
import yaml

from psiaudio import util

from cftsdata.abr import load_abr_analysis
from cftsdata.summarize_abr import load_abr_waveforms


class BadFilenameException(Exception):

    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return f'Bad filename: {self.filename}'


# Parser for the filename generated by the CFTS launcher.
P_PSI_FILENAME = re.compile(
	'^(?P<datetime>\d{8}-\d{6}) '
	'(?P<experimenter>\w+) '
	'(?P<animal_id>[-\w]+) '
	'((?P<ear>left|right) )?(?P<note>.*) '
	'(?P<experiment_type>(?:abr|dpoae|efr|memr|inear|dpgram|dual_dpoae)(_\w+)?).*$'
)


def parse_psi_filename(filename):
    try:
        groups = P_PSI_FILENAME.match(filename.stem).groupdict()
        groups['datetime'] = dt.datetime.strptime(groups['datetime'], '%Y%m%d-%H%M%S')
        groups['date'] = pd.to_datetime(groups['datetime'].date())
        groups['time'] = groups['datetime'].time()
        return groups
    except AttributeError:
        raise ValueError(f'Could not parse {filename.stem}')


def load_efr_level(filename, which='total'):
    result = pd.read_csv(filename, index_col=['fm', 'fc', 'frequency']) \
        ['level (dB SPL)'].rename('level').reset_index()
    if which == 'total':
        result = result.query('frequency == "total"').drop(columns=['frequency'])
    elif which == 'harmonics':
        result = result.query('frequency != "total"')
        result['frequency'] = result['frequency'].astype('f')
    else:
        raise ValueError(f'Unrecognized parameter for which: "{which}"')
    return result


def load_efr_harmonics(x):
    df = pd.read_csv(x)
    # This corrects for a difference in the data saved between the Bharadwaj
    # and Verhulst bootstrapping approaches. In Bharadwaj output, the harmonic
    # is the harmonic number (i.e., not the actual harmonic frequency). In the
    # Verhulst output, we only save the frequency, not the harmonic number.
    if 'frequency' not in df:
        df['frequency'] = df['harmonic']
        df['harmonic'] = df.eval('frequency / fm').astype('i')
    grouping = ['fc', 'fm', 'frequency', 'harmonic']
    result = df.groupby(grouping).mean().reset_index()
    if 'bootstrap' in result:
        return result.drop(columns='bootstrap')
    else:
        return result


def load_efr(x):
    result = pd.read_csv(x) \
        .groupby(['fc', 'fm']) \
        .mean().reset_index()
    if 'bootstrap' in result:
        return result.drop(columns='bootstrap')
    else:
        return result


def load_memr(filename, repeat=None):
    df = pd.read_csv(filename, index_col=['repeat', 'elicitor_level', 'frequency'])['amplitude']
    if repeat is not None:
        df = df.loc[repeat]
    return df.reset_index()


def load_memr_amplitude(filename, repeat=None, span=None):
    df = pd.read_csv(filename, index_col=['repeat', 'elicitor_level', 'span'])['amplitude']
    if repeat is not None:
        df = df.loc[repeat]
    if span is not None:
        df = df.xs(span, level='span')
    return df.reset_index()


def coerce_frequency(columns, octave_step, si_prefix=''):
    '''
    Decorator for methods in subclasses of `Dataset` that coerce frequencies in
    the DataFrame to the nearest octave.

    Parameters
    ----------
    columns : {str, list of string}
        Name or names of columns containing frequencies to coerce.
    octave_step : float
        Octave step to coerce frequencies.
    si_prefix : {'', 'k', list of string}
        Ensure that coercion is based on expected scale given SI unit. If list,
        must be one of the valid choices listed above. See
        `psiaudio.util.nearest_octave` for more information.
    '''
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(si_prefix, str):
        si_prefix = [si_prefix] * len(columns)
    if len(columns) != len(si_prefix):
        raise ValueError('Length of si_prefix should match length of columns')

    def inner(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            for c, si in zip(columns, si_prefix):
                try:
                    result[c] = util.nearest_octave(result[c], octave_step, si)
                except KeyError as e:
                    cols = ', '.join(result.columns)
                    raise KeyError(f'No column named "{c}". Valid columns are {cols}.') from e
            return result
        return wrapper
    return inner


class Dataset:

    def __init__(self, ephys_path=None, subpath=None):
        if ephys_path is None:
            ephys_path = os.environ['PROC_DATA_DIR']
        self.ephys_path = Path(ephys_path)
        self.subpath = subpath
        if subpath is not None:
            self.ephys_path = self.ephys_path / subpath
        if not self.ephys_path.exists():
            raise ValueError(f'Unknown data path {self.ephys_path}')
        self.raw_ephys_path = self.ephys_path

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
        return self.load_raw(lambda x: {}, filename_parser=parse_psi_filename,
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
        if should_load_cb is None:
            should_load_cb = lambda x: True
        result = []
        for filename in data_path.glob(glob):
            try:
                if '_exclude' in str(filename):
                    continue
                if '.imaris_cache' in str(filename):
                    continue
                if not should_load_cb(filename):
                    continue
                data = cb(filename)
                info = filename_parser(filename)
                if include_dataset:
                    info['dataset'] = filename.parent

                if info_as_cols:
                    for k, v in info.items():
                        if k in data:
                            raise ValueError('Column will get overwritten')
                        data[k] = v
                else:
                    data = pd.concat([data], keys=[tuple(info.values())], names=list(info.keys()))

                result.append(data)
            except Exception as e:
                raise ValueError(f'Error processing {filename}') from e
        if len(result) == 0:
            raise ValueError('No data found')
        if isinstance(data, pd.DataFrame):
            df = pd.concat(result)
        else:
            df = pd.DataFrame(result)
        return df

    def load_iec_psd(self, **kwargs):
        return self.load(
            lambda x: pd.read_csv(x),
            '**/*inear_speaker_calibration_chirp psd.csv',
            parse_psi_filename,
            **kwargs
        )

    def load_dpoae_io(self, **kwargs):
        return self.load(lambda x: pd.read_csv(x),
                          '**/*dpoae_io io.csv',
                          parse_psi_filename, **kwargs)

    def load_dpoae_th(self, criterion=None, **kwargs):
        def _load_dpoae_th(x):
            df = pd.read_csv(x, index_col=0)
            df.columns = df.columns.astype('f')
            if criterion is not None:
                df = df.loc[:, [criterion]]
            df.columns.name = 'criterion'
            return df.stack().rename('threshold').reset_index()

        return self.load(_load_dpoae_th,
                          '**/*dpoae_io th.csv',
                          parse_psi_filename, **kwargs)

    def load_dpgram(self, **kwargs):
        '''
        Load DPgram

        Returns
        -------
        df : pandas DataFrame indexed by F2 frequency and level.
        '''
        return self.load(lambda x: pd.read_csv(x),
                          '**/*dpgram.csv',
                          parse_psi_filename, **kwargs)

    def load_abr_io(self, level=None, **kwargs):
        def _load_abr_io(x):
            freq, th, rater, peaks = load_abr_analysis(x)
            peaks = peaks.reset_index()
            peaks['frequency'] = freq
            peaks['rater'] = rater
            if level is not None:
                peaks = peaks.query(f'level == {level}')
            return peaks
        abr_io = self.load(_load_abr_io,
                            '**/*analyzed.txt',
                            parse_psi_filename,
                            **kwargs)
        abr_io['w1'] = abr_io.eval('p1_amplitude - n1_amplitude')
        return abr_io

    def load_abr_th(self, rater=None, **kwargs):
        def _load_abr_th(x):
            freq, th, rater, _ = load_abr_analysis(x)
            return pd.Series({'frequency': freq, 'threshold': th, 'rater': rater})
        glob = '**/*-analyzed.txt' if rater is None else f'**/*-{rater}-analyzed.txt'
        return self.load(_load_abr_th,
                          glob,
                          parse_psi_filename,
                          **kwargs)

    def load_abr_waveforms(self, frequency=None, level=None, **kwargs):
        def _load_abr_waveforms(x):
            nonlocal frequency
            nonlocal level
            df = load_abr_waveforms(x)
            try:
                if frequency is not None:
                    df = df.xs(frequency, level='frequency', drop_level=False)
                if level is not None:
                    df = df.xs(level, level='level', drop_level=False)
                return df.stack() \
                    .rename('signal') \
                    .reset_index() \
                    .rename(columns={'time': 'timepoint'})
            except KeyError:
                return pd.DataFrame()

        return self.load(_load_abr_waveforms,
                         '**/*ABR average waveforms.csv',
                         parse_psi_filename,
                         **kwargs)

    def load_abr_settings(self, **kwargs):
        def _load_abr_settings(x):
            return pd.Series(json.loads(x.read_text()))
        return self.load(_load_abr_settings,
                          '**/*ABR experiment settings.json',
                          parse_psi_filename,
                          **kwargs)

    def load_abr_frequencies(self, **kwargs):
        '''
        Loads ABR frequencies

        Returns
        -------
        frequencies : pd.DataFrame
            Dataframe with one row per frequency.
        '''
        def _load_abr_frequencies(x):
            with x.open() as fh:
                frequencies = set(float(v) for v in fh.readline().split(',')[1:])
                return pd.Series({'frequencies': sorted(frequencies)})
        ds = self.load(_load_abr_frequencies,
                        '**/*ABR average waveforms.csv',
                        parse_psi_filename, **kwargs)
        result = []
        for _, row in ds.iterrows():
            for frequency in row['frequencies']:
                new_row = row.copy()
                del new_row['frequencies']
                new_row['frequency'] = frequency
                result.append(new_row)
        return pd.DataFrame(result)

    def load_abr_eeg_spectrum(self, max_freq=5e3, **kwargs):
        return self.load(lambda x: pd.read_csv(x),
                          '**/*ABR eeg spectrum.csv',
                          parse_psi_filename, **kwargs) \
            .query('frequency <= @max_freq')

    def load_abr_eeg_rms(self, **kwargs):
        def _load_abr_eeg_rms(x):
            return pd.Series(json.loads(x.read_text()))
        return self.load(_load_abr_eeg_rms,
                          '**/*ABR eeg rms.json',
                          parse_psi_filename, **kwargs)

    def _check_efr_params(self, efr_type, method):
        if efr_type.lower() not in ('sam', 'ram'):
            raise ValueError(f'Unrecognized EFR type "{efr_type}". Valid types are SAM or RAM.')
        if method.lower() not in ('verhulst', 'bharadwaj'):
            raise ValueError(f'Unrecognized method "{method}". Valid methods are Verhulst or Bharadwaj.')

    def load_efr_harmonics(self, efr_type, method, **kwargs):
        '''
        Load EFR harmonics

        Parameters
        ---------
        efr_type : {'SAM', 'RAM'}
            EFR to load harmonics for
        method : {'Verhulst', 'Bharadwaj'}
            Method for calculating EFR amplitude
        '''
        self._check_efr_params(efr_type, method)
        suffix = ' linear' if method.lower() == 'verhulst' else ''
        glob = f'**/*efr_{efr_type.lower()}*EFR harmonics{suffix}.csv'
        return self.load(load_efr_harmonics, glob, parse_psi_filename, **kwargs)

    def load_efr(self, efr_type, method, **kwargs):
        self._check_efr_params(efr_type, method)
        suffix = ' amplitude linear' if method.lower() == 'verhulst' else ''
        glob = f'**/*efr_{efr_type.lower()}*EFR{suffix}.csv'
        return self.load(load_efr, glob, parse_psi_filename, **kwargs)

    def load_efr_sam_level(self, **kwargs):
        return self.load(load_efr_level,
                         '**/*efr_sam*stimulus levels.csv',
                         parse_psi_filename, **kwargs)

    def load_efr_ram_level(self, **kwargs):
        return self.load(load_efr_level,
                         '**/*efr_ram*stimulus levels.csv',
                         parse_psi_filename, **kwargs)

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

    def load_memr(self, memr, repeat=None, total=False, **kwargs):
        etype = self._get_memr_etype(memr)
        glob = f'**/*{etype} MEMR_total.csv' if total \
            else f'**/*{etype} MEMR.csv'
        return self.load(partial(load_memr, repeat=repeat),
                         glob, parse_psi_filename, **kwargs)

    def load_memr_amplitude(self, memr, method='HT2', amplitude_type='raw'):
        '''
        Load the MEMR amplitude as calcuated by the given method
        '''
        etype = self._get_memr_etype(memr)
        glob = f'**/*{etype} {method} {amplitude_type} amplitude.csv'
        return self.load(lambda x: pd.read_csv(x), glob, parse_psi_filename)

    def load_memr_metrics(self, memr, method='HT2'):
        etype = self._get_memr_etype(memr)
        glob = f'**/*{etype} {method} threshold.json'
        return self.load(lambda x: json.loads(x.read_text()), glob,
                         parse_psi_filename)

    def load_memr_level(self, memr):
        etype = self._get_memr_etype(memr)
        glob = f'**/*{etype} raw_levels.csv'
        return self.load(lambda x: pd.read_csv(x), glob, parse_psi_filename)
