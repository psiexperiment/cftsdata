'''
Interface for reading tone-evoked ABR data generated by psiexperiment

Example
-------

.. code-block:: python

    from psidata.api import abr
    filename = '20220601-1200 ID1234 abr_io'
    with abr.load(filename) as fh:
        epochs = fh.get_epochs()
        epochs_filtered = fh.get_epochs_filtered()

'''
import logging
log = logging.getLogger(__name__)

from enum import Enum
from functools import lru_cache, partialmethod, wraps
import json
import os.path
from pathlib import Path
import shutil
import re
from glob import glob
import hashlib
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import signal

from psidata.api import Recording


# Max size of LRU cache
MAXSIZE = 1024


MERGE_PATTERN = \
    r'\g<date>-* ' \
    r'\g<experimenter> ' \
    r'\g<animal> ' \
    r'\g<ear> ' \
    r'\g<note> ' \
    r'\g<experiment>*'


def freq_to_label(freq):
    if freq == 'click':
        return 'Click'
    return f'{freq:.0f} Hz'


class ABRStim(Enum):
    '''
    ABR frequencies are typically reported as floats. To enable consistency
    with other types of ABRs, we map the stimulus type to an enum value.
    '''
    CLICK = -1


class ABRFile(Recording):
    '''
    Wrapper around an ABR file with methods for loading and querying data

    Parameters
    ----------
    base_path : string
        Path to folder containing ABR data
    '''

    def __init__(self, base_path, setting_table='erp_metadata'):
        super().__init__(base_path, setting_table)
        try:
            getattr(self, 'eeg')
        except AttributeError:
            raise ValueError('EEG data missing')
        try:
            getattr(self, 'erp_metadata')
        except AttributeError:
            raise ValueError('ERP metadata missing')

    @property
    def stimulus_type(self):
        '''
        Stimulus type used for ABR

        Returns
        -------
        stimulus_type
            'click' or 'tone' depending on stimulus used.
        '''
        return 'click' if 'abr_io_click' in self.base_path.stem else 'tone'

    @property
    @lru_cache(maxsize=MAXSIZE)
    def eeg(self):
        '''
        Continuous EEG signal in `BcolzSignal` format.
        '''
        if 'eeg' in self.carray_names:
            # Load and ensure that the EEG data is fine. If not, repair it and
            # reload the data.
            rootdir = self.base_path / 'eeg'
            if (rootdir / '__attrs__').exists():
                import bcolz
                from .bcolz_tools import repair_carray_size
                eeg = bcolz.carray(rootdir=rootdir)
                if len(eeg) == 0:
                    log.debug('EEG for %s is corrupt. Repairing.', self.base_path)
                    repair_carray_size(rootdir)
        return self.__getattr__('eeg')

    @property
    @lru_cache(maxsize=MAXSIZE)
    def erp_metadata(self):
        '''
        Raw ERP metadata in DataFrame format

        There will be one row for each epoch and one column for each parameter
        from the ABR experiment. For simplicity, all parameters beginning with
        `target_tone_` have that string removed. For example,
        `target_tone_frequency` will become `frequency`).
        '''
        data = self.__getattr__('erp_metadata')
        if self.stimulus_type == 'click':
            data = data.drop(columns=['duration'])
        return data \
            .rename(columns=lambda x: x.replace('target_tone_', '')) \
            .rename(columns=lambda x: x.replace('target_click_', ''))

    def get_epochs(self, offset=0, duration=8.5e-3, detrend='constant',
                   downsample=None, reject_threshold='saved',
                   reject_mode='absolute', columns='auto', averages='saved',
                   cb=None, signal='eeg'):
        '''
        Extract event-related epochs from EEG

        Parameters
        ----------
        {common_docstring}
        {epochs_docstring}
        '''
        add_frequency = False
        if self.stimulus_type != 'tone':
            if 'frequency' in columns:
                columns = columns[:]
                columns.remove('frequency')
                add_frequency = True

        fn = getattr(self, signal).get_epochs
        result = fn(self.erp_metadata, offset, duration, detrend,
                    downsample=downsample, columns=columns, cb=cb)
        result = self._apply_reject(result, reject_threshold, reject_mode)
        result = self._apply_n(result, averages)
        if add_frequency:
            result = pd.concat([result], [self.stimulus_type], names=['frequency'])

        return result

    def get_random_segments(self, n, offset=0, duration=8.5e-3,
                            detrend='constant', downsample=None,
                            reject_threshold='saved', reject_mode='absolute',
                            cb=None):
        '''
        Extract random segments from filtered EEG

        Parameters
        ----------
        n : int
            Number of segments to return
        {common_docstring}
        '''
        fn = self.eeg.get_random_segments
        result = fn(n, offset, duration, detrend, downsample=downsample, cb=cb)
        return self._apply_reject(result, reject_threshold, reject_mode)

    def get_epochs_filtered(self, filter_lb=300, filter_ub=3000,
                            filter_order=1, offset=-1e-3, duration=10e-3,
                            detrend='constant', pad_duration=10e-3,
                            downsample=None, reject_threshold='saved',
                            reject_mode='absolute', columns='auto',
                            averages='saved', cb=None):
        '''
        Extract event-related epochs from filtered EEG

        Parameters
        ----------
        {filter_docstring}
        {common_docstring}
        {epochs_docstring}
        '''
        add_frequency = False
        if self.stimulus_type != 'tone':
            if 'frequency' in columns:
                columns = columns[:]
                columns.remove('frequency')
                add_frequency = True

        fn = self.eeg.get_epochs_filtered
        result = fn(md=self.erp_metadata, offset=offset, duration=duration,
                    filter_lb=filter_lb, filter_ub=filter_ub,
                    filter_order=filter_order, detrend=detrend,
                    pad_duration=pad_duration, downsample=downsample,
                    columns=columns, cb=cb)
        result = self._apply_reject(result, reject_threshold, reject_mode)
        result = self._apply_n(result, averages)
        if add_frequency:
            result = pd.concat([result], keys=[self.stimulus_type], names=['frequency'])

        return result

    def get_random_segments_filtered(self, n, filter_lb=300, filter_ub=3000,
                                     filter_order=1, offset=-1e-3,
                                     duration=10e-3, detrend='constant',
                                     pad_duration=10e-3,
                                     downsample=None,
                                     reject_threshold='saved',
                                     reject_mode='absolute', cb=None):
        '''
        Extract random segments from EEG

        Parameters
        ----------
        n : int
            Number of segments to return
        {filter_docstring}
        {common_docstring}
        '''
        fn = self.eeg.get_random_segments_filtered
        result = fn(n, offset, duration, filter_lb, filter_ub, filter_order,
                    detrend, pad_duration, downsample=downsample, cb=cb)
        return self._apply_reject(result, reject_threshold, reject_mode)

    def _apply_reject(self, result, reject_threshold, reject_mode):
        result = result.dropna()

        if reject_threshold is None:
            return result
        if reject_threshold is np.inf:
            return result
        if reject_threshold == 'saved':
            # 'reject_mode' wasn't added until a later version of the ABR
            # program, so we set it to the default that was used before if not
            # present.
            reject_threshold = self.get_setting('reject_threshold')
            reject_mode = self.get_setting_default('reject_mode', 'absolute')

        if reject_mode == 'absolute':
            m = (result < reject_threshold).all(axis=1)
            result = result.loc[m]
        elif reject_mode == 'amplitude':
            # TODO
            raise NotImplementedError

        return result

    def _apply_n(self, result, averages):
        '''
        Limit epochs to the specified number of averages
        '''
        if averages is None:
            return result
        if averages is np.inf:
            return result
        if averages == 'saved':
            averages = self.get_setting('averages')

        grouping = list(result.index.names)
        grouping.remove('t0')
        if 'polarity' in result.index.names:
            n = averages // 2
            if (n * 2) != averages:
                m = f'Number of averages {averages} not divisible by 2'
                raise ValueError(m)
        else:
            n = averages
        return result.groupby(grouping, group_keys=False) \
            .apply(lambda x: x.iloc[:n])


class ABRSupersetFile:

    def __init__(self, *base_paths):
        self._fh = [ABRFile(base_path) for base_path in base_paths]

    def _merge_results(self, fn_name, *args, merge_on_file=False, **kwargs):
        result_set = [getattr(fh, fn_name)(*args, **kwargs) for fh in self._fh]
        if merge_on_file:
            return pd.concat(result_set, keys=range(len(self._fh)), names=['file'])
        offset = 0
        for result in result_set:
            t0 = result.index.get_level_values('t0')
            if offset > 0:
                result.index = result.index.set_levels(t0 + offset, 't0')
            offset += t0.max() + 1
        return pd.concat(result_set)

    get_epochs = partialmethod(_merge_results, 'get_epochs')
    get_epochs_filtered = partialmethod(_merge_results, 'get_epochs_filtered')
    get_random_segments = partialmethod(_merge_results, 'get_random_segments')
    get_random_segments_filtered = \
        partialmethod(_merge_results, 'get_random_segments_filtered')

    @classmethod
    def from_pattern(cls, base_path):
        head, tail = os.path.split(base_path)
        glob_tail = FILE_RE.sub(MERGE_PATTERN, tail)
        glob_pattern = os.path.join(head, glob_tail)
        folders = glob(glob_pattern)
        inst = cls(*folders)
        inst._base_path = base_path
        return inst

    @classmethod
    def from_folder(cls, base_path):
        folders = [os.path.join(base_path, f) \
                   for f in os.listdir(base_path)]
        inst = cls(*[f for f in folders if os.path.isdir(f)])
        inst._base_path = base_path
        return inst

    @property
    def erp_metadata(self):
        result_set = [fh.erp_metadata for fh in self._fh]
        return pd.concat(result_set, keys=range(len(self._fh)), names=['file'])


def list_abr_experiments(base_path):
    if is_abr_experiment(base_path, allow_superset=False):
        return [base_path]

    experiments = []
    base_path = Path(base_path)
    if base_path.is_file():
        return experiments

    for path in Path(base_path).iterdir():
        if path.is_dir():
            experiments.extend(list_abr_experiments(path))
    return experiments


def load(base_path, allow_superset=False):
    '''
    Load ABR data

    Parameters
    ----------
    base_path : string
        Path to folder
    allow_superset : bool
        If True, will merge all subfolders containing valid ABR data into a
        single superset ABR file. If False, base_path must be a valid ABR
        dataset.

    Returns
    -------
    {ABRFile, ABRSupersetFile}
        Depending on folder, will return either an instance of `ABRFile` or
        `ABRSupersetFile`.
    '''
    if allow_superset:
        raise NotImplementedError
    return ABRFile(base_path)


def is_abr_experiment(base_path, allow_superset=False):
    '''
    Checks if path contains valid ABR data

    Parameters
    ----------
    base_path : string
        Path to folder

    Returns
    -------
    bool
        True if path contains valid ABR data, False otherwise. If path doesn't
        exist, False is returned.
    '''
    try:
        result = load(base_path, allow_superset)
        return True
    except Exception as e:
        return False


P_TH = re.compile(r'Threshold \(dB SPL\): ([\w.-]+)')
P_FREQ = re.compile(r'Frequency \(kHz\): ([\d.]+)')
P_FILENAME = re.compile(r'.*-(\d+\.\d+|click)(?:kHz)?-(?:(\w+)-)?analyzed.txt')

def load_abr_analysis(filename,
                      freq_from_filename=True,
                      subthreshold_handling='discard'):
    '''
    Load ABR analysis from file

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Name of file to load.
    freq_from_filename : bool
        If True, load frequency from filename. The frequency stored in the
        header of the file rounded to the nearest 10 Hz, but the frequency in
        the filename is not.

    Returns
    -------

    '''
    rename = {
        'Level': 'level',
        '1msec Avg': 'baseline',
        '1msec StDev': 'baseline_std',
        'P1 Latency': 'p1_latency',
        'P1 Amplitude': 'p1_amplitude',
        'N1 Latency': 'n1_latency',
        'N1 Amplitude': 'n1_amplitude',
        'P2 Latency': 'p2_latency',
        'P2 Amplitude': 'p2_amplitude',
        'N2 Latency': 'n2_latency',
        'N2 Amplitude': 'n2_amplitude',
        'P3 Latency': 'p3_latency',
        'P3 Amplitude': 'p3_amplitude',
        'N3 Latency': 'n3_latency',
        'N3 Amplitude': 'n3_amplitude',
        'P4 Latency': 'p4_latency',
        'P4 Amplitude': 'p4_amplitude',
        'N4 Latency': 'n4_latency',
        'N4 Amplitude': 'n4_amplitude',
        'P5 Latency': 'p5_latency',
        'P5 Amplitude': 'p5_amplitude',
        'N5 Latency': 'n5_latency',
        'N5 Amplitude': 'n5_amplitude',
    }

    filename = Path(filename)

    with filename.open() as fh:
        for line in fh:
            # Parse the threshold string
            if line.startswith('Threshold'):
                th_string = P_TH.search(line).group(1)
                if th_string == 'None':
                    th = -np.inf
                elif th_string == 'inf':
                    th = np.inf
                elif th_string == '-inf':
                    th = -np.inf
                else:
                    th = float(th_string)

            if line.startswith('Frequency'):
                freq = float(P_FREQ.search(line).group(1))*1e3

            if line.startswith('NOTE'):
                break

        data = pd.io.parsers.read_csv(fh, sep='\t')
        data.rename(columns=rename, inplace=True)

    keep_cols = list(rename.values())
    keep = [c for c in data.columns if c in keep_cols]
    data = data[keep]


    # Discard all sub-threshold data
    if subthreshold_handling == 'discard':
        m = data['level'] >= th
        data = data.loc[m]
    elif subthreshold_handling == 'zero':
        m = data['level'] < th
        update_cols = keep[:]
        update_cols.remove('baseline')
        update_cols.remove('baseline_std')
        update_cols.remove('level')
        data.loc[m, update_cols] = 0
    elif subthreshold_handling == 'nan':
        m = data['level'] < th
        update_cols = keep[:]
        update_cols.remove('baseline')
        update_cols.remove('baseline_std')
        update_cols.remove('level')
        data.loc[m, update_cols] = np.nan

    data = data.set_index('level', verify_integrity=True).sort_index()

    try:
        filename_freq, rater = P_FILENAME.match(filename.name).groups()
        if freq_from_filename:
            if filename_freq == 'click':
                freq = 'click'
            else:
                freq = float(filename_freq) * 1e3
    except AttributeError:
        raise ValueError(f'Could not parser rater and frequency from {filename.name}')

    return freq, th, rater, data


filter_docstring = '''
        filter_lb : float
            Lower bound of filter passband, in Hz.
        filter_ub : float
            Upper bound of filter passband, in Hz.
        filter_order : int
            Filter order. Note that the effective order will be double this
            since we use zero-phase filtering.
'''.strip()


common_docstring = '''
        offset : float
            Starting point of epoch, in seconds re. trial start. Can be
            negative to capture prestimulus baseline.
        duration : float
            Duration of epoch, in seconds, relative to offset.
        detrend : {'constant', 'linear', None}
            Method for detrending
        pad_duration : float
            Duration, in seconds, to pad epoch prior to filtering. The extra
            samples will be discarded after filtering.
        reject_threshold : {None, 'saved', float}
            Rejects epochs according to the following criteria:
                * `None`: do not reject trials
                * 'saved': Use the value stored in the file
                * float: Use the provided value.
        reject_mode : string
            Not imlemented
        cb : {None, callable}
            If a callable is provided, this will be called with the current
            fraction of segments loaded from the file. This is useful when
            loading many segments over a slow connection.
        averages : {None, 'saved', int}
            Returnes the desired number of epochs (after rejection) according
            to the following criteria:
                * `None`: return all trials that pass the rejection filter
                * 'saved': use the value stored in the file to determine the
                  number of trials to return
                * int: Use the provided value.
'''.strip()


epochs_docstring = '''
        columns : {'auto', list of names}
            Columns to include
        averages : None
            Limits the number of epochs returned to the number of averages
            specified. If None, use the value stored in the file. Otherwise,
            use the provided value. To return all epochs, use `np.inf`. For
            dual-polarity data, care will be taken to ensure the number of
            trials from each polarity match (even when set to `np.inf`).
'''.strip()


def format_docstrings(klass):
    fmt = {
        'common_docstring': common_docstring,
        'filter_docstring': filter_docstring,
        'epochs_docstring': epochs_docstring,
    }
    for member_name in dir(klass):
        member = getattr(klass, member_name)
        try:
            member.__doc__ = member.__doc__.format(**fmt)
        except:
            pass


format_docstrings(ABRFile)
