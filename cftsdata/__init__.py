'''
This module provides classes and functions that facilitate working with
recordings created by psiexperiment.

The base class of all experiments is `Recording`. Some subclasses (e.g.,
`psi.data.io.abr.ABRFile`) offer more specialized support for a particular
experiment type.
'''
import logging
log = logging.getLogger(__name__)

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


def get_unique_columns(df, exclude=None):
    return [c for c in df if (len(df[c].unique()) > 1) and (c not in exclude)]


class Recording:
    '''
    Wrapper around a recording created by psiexperiment

    Parameters
    ----------
    base_path : :obj:`str` or :obj:`pathlib.Path`
        Folder containing recordings

    Attributes
    ----------
    base_path : pathlib.Path
        Folder containing recordings
    carray_names : set
        List of Bcolz carrays in this recording
    ctable_names : set
        List of Bcolz ctables in this recording
    ttable_names : set
        List of CSV-formatted tables in this recording

    The `__getattr__` method is implemented to allow accessing arrays and
    tables by name. For example, if you have a ctable called `erp_metadata`:

        recording = Recording(base_path)
        erp_md = recording.erp_metadata

    When using this approach, all tables are loaded into memory and returned as
    instances of `pandas.DataFrame`. All arrays are returned as `Signal`
    instances. Signal instances do not load the data into memory until the data
    is requested.
    '''

    #: Mapping of names for CSV-formatted table to a list of columns that
    #: should be used as indices. For example:
    #:     {'tone_sens': ['channel_name', 'frequency']}
    #: This attribute is typically used by subclasses to automate handling of
    #: loading tables into DataFrames.
    _ttable_indices = {}

    def __init__(self, base_path):
        bp = Path(base_path)
        self.base_path = bp
        self._refresh_names()

    def _refresh_names(self):
        '''
        Utility function to refresh list of signals and tables

        This is primarily used by repair functions that may need to fix various
        items in the folder when the class is first created.
        '''
        bp = self.base_path
        self.carray_names = {d.parent.stem for d in bp.glob('*/meta')}
        self.ctable_names = {d.parent.parent.stem for d in bp.glob('*/*/meta')}
        self.ttable_names = {d.stem for d in bp.glob('*.csv')}

    def __getattr__(self, attr):
        if attr in self.carray_names:
            return self._load_bcolz_signal(attr)
        if attr in self.ctable_names:
            return self._load_bcolz_table(attr)
        elif attr in self.ttable_names:
            return self._load_text_table(attr)
        raise AttributeError

    def __repr__(self):
        lines = [f'Recording at {self.base_path.name} with:']
        lines.append(f'* Bcolz carrays {self.carray_names}')
        lines.append(f'* Bcolz ctables {self.ctable_names}')
        lines.append(f'* CSV tables {self.ttable_names}')
        return '\n'.join(lines)

    @functools.lru_cache()
    def _load_bcolz_signal(self, name):
        from .bcolz_tools import BcolzSignal
        return BcolzSignal(self.base_path / name)

    @functools.lru_cache()
    def _load_bcolz_table(self, name):
        from .bcolz_tools import load_ctable_as_df
        return load_ctable_as_df(self.base_path / name)

    @functools.lru_cache()
    def _load_text_table(self, name):
        import pandas as pd
        path = (self.base_path / name).with_suffix('.csv')
        if path.stat().st_size == 0:
            return pd.DataFrame()
        index_col = self._ttable_indices.get(name, None)
        df = pd.read_csv(path, index_col=index_col)
        drop = [c for c in df.columns if c.startswith('Unnamed:')]
        return df.drop(columns=drop)


class Signal:

    def get_epochs(self, md, offset, duration, detrend=None, columns='auto'):
        fn = self.get_segments
        return self._get_epochs(fn, md, offset, duration, detrend=detrend,
                                columns=columns)

    def get_epochs_filtered(self, md, offset, duration, filter_lb, filter_ub,
                            filter_order=1, detrend='constant',
                            pad_duration=10e-3, columns='auto'):
        fn = self.get_segments_filtered
        return self._get_epochs(fn, md, offset, duration, filter_lb, filter_ub,
                                filter_order, detrend, pad_duration,
                                columns=columns)

    def _get_epochs(self, fn, md, *args, columns='auto', **kwargs):
        if columns == 'auto':
            columns = get_unique_columns(md, exclude=['t0'])
        t0 = md['t0'].values
        arrays = [md[c] for c in columns]
        arrays.append(t0)
        df = fn(t0, *args, **kwargs)
        df.index = pd.MultiIndex.from_arrays(arrays, names=columns + ['t0'])
        return df

    def get_segments(self, times, offset, duration, detrend=None):
        times = np.asarray(times)
        indices = np.round((times + offset) * self.fs).astype('i')
        samples = round(duration * self.fs)

        m = (indices >= 0) & ((indices + samples) < self.shape[-1])
        if not m.all():
            i = np.flatnonzero(~m)
            log.warn('Missing epochs %d', i)

        values = np.concatenate([self[i:i+samples][np.newaxis] \
                                 for i in indices[m]])
        if detrend is not None:
            values = signal.detrend(values, axis=-1, type=detrend)

        t = np.arange(samples)/self.fs + offset
        columns = pd.Index(t, name='time')
        index = pd.Index(times[m], name='t0')
        df = pd.DataFrame(values, index=index, columns=columns)
        return df.reindex(times)

    def _get_segments_filtered(self, fn, offset, duration, filter_lb, filter_ub,
                               filter_order=1, detrend='constant',
                               pad_duration=10e-3):
        Wn = (filter_lb/(0.5*self.fs), filter_ub/(0.5*self.fs))
        b, a = signal.iirfilter(filter_order, Wn, btype='band', ftype='butter')
        df = fn(offset-pad_duration, duration+pad_duration, detrend)
        df[:] = signal.filtfilt(b, a, df.values, axis=-1)
        return df.loc[:, offset:offset+duration]

    def get_random_segments(self, n, offset, duration, detrend):
        t_min = -offset
        t_max = self.duration-duration-offset
        times = np.random.uniform(t_min, t_max, size=n)
        return self.get_segments(times, offset, duration, detrend)

    def get_segments_filtered(self, times, *args, **kwargs):
        fn = functools.partial(self.get_segments, times)
        return self._get_segments_filtered(fn, *args, **kwargs)

    def get_random_segments_filtered(self, n, *args, **kwargs):
        fn = functools.partial(self.get_random_segments, n)
        return self._get_segments_filtered(fn, *args, **kwargs)
