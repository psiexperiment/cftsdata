'''
Loaders for the data products written by `cftsdata.summarize_abr_adaptation`
'''
import json

import pandas as pd

from ..abr import ABRStim
from ._util import ensure_suffix


SUFFIXES = {
    'waveforms': 'ABR adaptation average waveforms.csv',
    'train_waveforms': 'ABR adaptation train waveforms.csv',
    'amplitude': 'ABR adaptation amplitude.csv',
    'trials': 'ABR adaptation trials.csv',
    'processing_settings': 'ABR adaptation processing settings.json',
    'settings': 'ABR adaptation experiment settings.json',
    'eeg_spectrum': 'ABR adaptation eeg spectrum.csv',
    'eeg_rms': 'ABR adaptation eeg rms.json',
}


def _code_click(df):
    '''
    Report clicks as `ABRStim.CLICK.value`, as `loaders.abr` does
    '''
    if 'frequency' not in df.index.names:
        return df
    index = df.index.to_frame(index=False)
    index['frequency'] = [ABRStim.CLICK.value if v == 'click' else float(v)
                          for v in index['frequency']]
    df.index = pd.MultiIndex.from_frame(index)
    return df


def _select(df, **levels):
    for name, value in levels.items():
        if value is not None:
            df = df.xs(value, level=name, drop_level=False)
    return df


def _read_waveforms(filename, index_col):
    df = pd.read_csv(filename, index_col=index_col)
    df.columns = df.columns.astype('float')
    df.columns.name = 'time'
    return _code_click(df)


@ensure_suffix(SUFFIXES['waveforms'])
def waveforms(filename, frequency=None, level=None, repeat=None):
    '''
    Load the average waveform of each repeat (i.e., burst in the train)

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the average waveforms file itself.
    frequency : {None, float}
        If provided, return only this frequency.
    level : {None, float}
        If provided, return only this level.
    repeat : {None, int}
        If provided, return only this repeat (counting from 0).

    Returns
    -------
    df : pd.DataFrame
        Waveforms indexed by frequency, level and repeat, one column per
        timepoint (in seconds re. burst onset). Pass through
        `cftsdata.util.to_long` for long format.
    '''
    df = _read_waveforms(filename, [0, 1, 2])
    return _select(df, frequency=frequency, level=level, repeat=repeat)


@ensure_suffix(SUFFIXES['train_waveforms'])
def train_waveforms(filename, frequency=None, level=None):
    '''
    Load the average waveform of the whole train

    Returns
    -------
    df : pd.DataFrame
        Waveforms indexed by frequency and level, one column per timepoint
        (in seconds re. the start of the train).
    '''
    df = _read_waveforms(filename, [0, 1])
    return _select(df, frequency=frequency, level=level)


@ensure_suffix(SUFFIXES['amplitude'])
def amplitude(filename, frequency=None, level=None):
    '''
    Load the amplitude of each repeat

    Returns
    -------
    df : pd.DataFrame
        Indexed by frequency, level and repeat. Columns are the start of the
        repeat re. the start of the train (`repeat_t0`, in seconds), the
        peak-to-peak (`ptp`) and RMS (`rms`) amplitude in the analysis window,
        and both normalized to the first repeat (`ptp_norm`, `rms_norm`).
    '''
    df = _code_click(pd.read_csv(filename, index_col=[0, 1, 2]))
    return _select(df, frequency=frequency, level=level)


@ensure_suffix(SUFFIXES['trials'])
def trials(filename):
    '''
    Load the number of trains averaged and the fraction rejected

    Returns
    -------
    df : pd.DataFrame
        Indexed by frequency and level with columns `epoch_n` (number of
        trains averaged) and `epoch_reject_ratio` (fraction of trains rejected
        as artifacts).
    '''
    return _code_click(pd.read_csv(filename, index_col=[0, 1]))


@ensure_suffix(SUFFIXES['settings'])
def settings(filename):
    '''
    Load the experiment settings used to acquire the dataset
    '''
    return pd.Series(json.loads(filename.read_text()))


@ensure_suffix(SUFFIXES['processing_settings'])
def processing_settings(filename):
    '''
    Load the settings used to summarize the dataset
    '''
    return pd.Series(json.loads(filename.read_text()))


@ensure_suffix(SUFFIXES['eeg_spectrum'])
def eeg_spectrum(filename, max_freq=5e3):
    '''
    Load the spectrum of the EEG recorded during the experiment
    '''
    return pd.read_csv(filename).query('frequency <= @max_freq')


@ensure_suffix(SUFFIXES['eeg_rms'])
def eeg_rms(filename):
    '''
    Load the RMS of the EEG recorded during the experiment
    '''
    return pd.Series(json.loads(filename.read_text()))
