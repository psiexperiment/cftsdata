import logging
log = logging.getLogger(__name__)

from functools import lru_cache

import numpy as np
import pandas as pd

from psidata.api import Recording

# Max size of LRU cache
MAXSIZE = 1024


def isodp_th(l2, dp, nf, criterion):
    '''
    Computes iso-DP threshold for a level sweep at a single frequency

    Parameters
    ----------
    l2 : array-like
        Requested F2 levels
    dp : array-like
        Measured DPOAE levels
    nf : array-like
        Measured DPOAE noise floor
    criterion : float
        Threshold criterion (e.g., value that the input-output function must
        exceed)

    Returns
    -------
    threshold : float
        If no threshold is identified, NaN is returned
    '''
    # First, discard up to the first level where the DPOAE exceeds one standard
    # deviation from the noisne floor
    nf_crit = np.mean(nf) + np.std(nf)
    i = np.flatnonzero(dp < nf_crit)
    if len(i):
        dp = dp[i[-1]:]
        l2 = l2[i[-1]:]

    # Now, loop through every pair of points until we find the first pair that
    # brackets criterion (for non-Python programmers, this uses chained
    # comparision operators and is not a bug)
    for l_lb, l_ub, d_lb, d_ub in zip(l2[:-1], l2[1:], dp[:-1], dp[1:]):
        if d_lb < criterion <= d_ub:
            return np.interp(criterion, [d_lb, d_ub], [l_lb, l_ub])
    return np.nan


def isodp_th_criterions(df, criterions=None, debug=False):
    '''
    Helper function that takes dataframe containing a single frequency and
    calculates threshold for each criterion.
    '''
    if criterions is None:
        criterions = [-5, 0, 5, 10, 15, 20, 25]

    if ':dB' in df.columns:
        # This is used for thresholding data already in EPL CFTS format
        l2 = df.loc[:, ':dB']
        dp = df.loc[:, '2f1-f2(dB)']
        nf = df.loc[:, '2f1-f2Nse(dB)']
    elif 'secondary_tone_level' in df.columns:
        l2 = df.loc[:, 'secondary_tone_level'].values
        dp = df.loc[:, 'dpoae_level'].values
        nf = df.loc[:, 'dpoae_noise_floor'].values
    elif 'f2_level' in df.columns:
        l2 = df.loc[:, 'f2_level'].values
        dp = df.loc[:, 'dp_level'].values
        nf = df.loc[:, 'dp_nf'].values
    else:
        raise ValueError('Could not find columns needed')

    th = [isodp_th(l2, dp, nf, c) for c in criterions]
    index = pd.Index(criterions, name='criterion')
    return pd.Series(th, index=index, name='threshold')


def dpoae_renamer(x):
    if x in ('f1_level', 'f2_level', 'dpoae_level'):
        return f'meas_{x}'
    if x in ('f2_frequency'):
        return f'req_{x}'
    return x.replace('primary_tone', 'f1') \
        .replace('secondary_tone', 'f2')


class DPOAEFile(Recording):

    def __init__(self, base_path, setting_table='dpoae_store'):
        super().__init__(base_path, setting_table)

    @property
    @lru_cache(maxsize=MAXSIZE)
    def results(self):
        data = getattr(self, 'dpoae_store')
        data = data.rename(columns=dpoae_renamer)

        # Add in the start/stop time of the actual stimulus itself. The
        # ts_start and ts_end timestamps indicate what was captured for the
        # online analysis.
        ts = self.event_log.query('event in ("dpoae_start", "experiment_end")')['timestamp'].values
        ts_start = ts[:-1]
        ts_end = ts[1:]
        if len(ts_start) != len(data):
            raise ValueError('Mismatch between event log and DPOAE metadata')
        data['dp_start'] = ts_start
        data['dp_end'] = ts_end
        return data


def load(filename):
    return DPOAEFile(filename)
