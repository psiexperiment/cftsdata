'''
Interface for reading tone-evoked ABR data generated by psiexperiment

This supports merging across aggregate files. Right now there's no support for
reading from the raw (i.e., untrial_filtered) data. Maybe eventually. It wouldn't be
hard to add, but would need support for trial_filter specs as well as ensuring that we
pull out enough pre/post samples for proper trial_filtering.
'''
import os.path
import shutil
import re
from glob import glob

import bcolz
import numpy as np
import pandas as pd
from scipy import signal

from .bcolz_tools import load_ctable_as_df


MERGE_PATTERN = \
    r'\g<date>-* ' \
    r'\g<experimenter> ' \
    r'\g<animal> ' \
    r'\g<ear> ' \
    r'\g<note> ' \
    r'\g<experiment>*'


def make_query(trials, base_name='target_tone_'):
    if base_name is not None:
        trials = trials.copy()
        trials = {'{}{}'.format(base_name, k): v for k, v in trials.items()}
    queries = ['({} == {})'.format(k, v) for k, v in trials.items()]
    return ' & '.join(queries)


def format_columns(columns, base_name):
    if columns is None:
        columns = ['t0']
        names = ['t0']
    else:
        names = columns + ['t0']
        if base_name is not None:
            columns = ['{}{}'.format(base_name, c) for c in columns]
        columns = columns + ['t0']
    return columns, names


class ABRFile:

    def __init__(self, base_folder):
        self._base_folder = base_folder
        self._eeg_folder = os.path.join(base_folder, 'eeg')
        self._erp_folder = os.path.join(base_folder, 'erp')
        self._erp_md_folder = os.path.join(base_folder, 'erp_metadata')
        self._eeg = bcolz.carray(rootdir=self._eeg_folder)
        self._erp = bcolz.carray(rootdir=self._erp_folder)
        self.trial_log = load_ctable_as_df(self._erp_md_folder)

    def get_epochs(self, offset=0, duration=8.5e-3, padding_samples=0,
                   detrend='constant', base_name='target_tone_', columns=None,
                   **trials):

        columns, names = format_columns(columns, base_name)
        query = make_query(trials, base_name)
        if query:
            result_set = self.trial_log.query(query)[columns]
        else:
            result_set = self.trial_log[columns]

        fs = self._eeg.attrs['fs']
        duration_samples = round(duration*fs)

        epochs = []
        index = []
        max_samples = self._eeg.shape[-1]

        for i, (_, row) in enumerate(result_set.iterrows()):
            t0 = row.t0
            lb = int(round((row.t0+offset)*fs))
            ub = lb + duration_samples
            lb -= padding_samples
            ub += padding_samples

            if lb < 0 or ub > max_samples:
                mesg = 'Data missing for epochs {} through {}'
                mesg = mesg.format(i+1, len(self.trial_log))
                print(mesg)
                break

            epoch = self._eeg[lb:ub]
            epochs.append(epoch[np.newaxis])
            index.append(row)

        n_samples = len(epoch)
        t = (np.arange(n_samples)-padding_samples)/self.fs + offset
        epochs = np.concatenate(epochs, axis=0)

        if detrend is not None:
            epochs = signal.detrend(epochs, -1, detrend)

        index = pd.MultiIndex.from_tuples(index, names=names)
        df = pd.DataFrame(epochs, columns=t, index=index)
        df.sort_index(inplace=True)
        return df

    def get_epochs_filtered(self, filter_lb=300, filter_ub=3000,
                            filter_order=1, offset=-1e3, duration=10e-3,
                            detrend='constant', pad_duration=10e-3,
                            base_name='target_tone_', columns=None, **trials):

        Wn = (filter_lb/self.fs, filter_ub/self.fs)
        b, a = signal.iirfilter(filter_order, Wn, btype='band', ftype='butter')
        padding_samples = round(pad_duration*self.fs)
        df = self.get_epochs(offset, duration, padding_samples, detrend,
                             base_name, columns, **trials)

        # The vectorized approach takes up too much memory on some of the older
        # NI PXI systems. Hopefully someday we can move away from all this!
        epochs_filtered = []
        for epoch in df.values:
            e = signal.filtfilt(b, a, epoch)
            e = e[padding_samples:-padding_samples]
            epochs_filtered.append(e)

        columns = df.columns[padding_samples:-padding_samples]
        return pd.DataFrame(epochs_filtered, index=df.index, columns=columns)

    def get_epoch_groups(self, *columns):
        groups = self.count_epochs(*columns)
        results = []
        keys = []
        for index, _ in groups.iteritems():
            trial_filter = {c: i for c, i in zip(columns, index)}
            epochs = self.get_epochs(trial_filter)
            keys.append(index)
            results.append(epochs)
        index = pd.MultiIndex.from_tuples(keys, names=columns)
        return pd.Series(results, index=index, name='epochs')

    @property
    def fs(self):
        return self._eeg.attrs['fs']


class ABRSupersetFile:

    def __init__(self, *base_folders):
        self._fh = [ABRFile(base_folder) for base_folder in base_folders]

    def get_epochs_filtered(self, *args, **kwargs):
        epoch_set = []
        keys = []
        for fh in self._fh:
            epochs = fh.get_epochs_filtered(*args, **kwargs)
            keys.append(os.path.basename(fh._base_folder))
            epoch_set.append(epochs)
        return pd.concat(epoch_set, keys=keys, names=['file'])

    @classmethod
    def from_pattern(cls, base_folder):
        head, tail = os.path.split(base_folder)
        glob_tail = FILE_RE.sub(MERGE_PATTERN, tail)
        glob_pattern = os.path.join(head, glob_tail)
        folders = glob(glob_pattern)
        cls(*folders)

    @classmethod
    def from_folder(cls, base_folder):
        folders = os.listdir(base_folder)
        folders = [os.path.join(base_folder, f) for f in folders]
        return cls(*folders)

    @property
    def fs(self):
        fs = [fh.fs for fh in self._fh]
        if len(set(fs)) != 1:
            raise ValueError('Sampling rate of ABR sets differ')
        return fs[0]

    @property
    def trial_log(self):
        return pd.concat([fh.trial_log for fh in self._fh], ignore_index=True)


def load(base_folder):
    check = os.path.join(base_folder, 'erp')
    if os.path.exists(check):
        return ABRFile(base_folder)
    else:
        return ABRSupersetFile.from_folder(base_folder)
