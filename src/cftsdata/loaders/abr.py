'''
Loaders for the data products written by `cftsdata.summarize_abr`
'''
import json
from pathlib import Path

import pandas as pd

from ..abr import ABRStim, load_abr_analysis
from ._util import ensure_suffix


SUFFIXES = {
    'waveforms': 'ABR average waveforms.csv',
    'processing_settings': 'ABR processing settings.json',
    'settings': 'ABR experiment settings.json',
    'eeg_spectrum': 'ABR eeg spectrum.csv',
    'eeg_rms': 'ABR eeg rms.json',
    'abrpresto_th': 'ABRpresto threshold.json',
}

# Peak analyses are saved by the peak-picking program, one file per frequency
# and rater, so they are found by glob rather than by a single suffix. Click
# analyses are named `-click-` rather than `-{freq}kHz-`, so the glob must not
# require the unit.
P_ANALYZED = '*-analyzed.txt'


@ensure_suffix(SUFFIXES['waveforms'])
def waveforms(filename, frequency=None, level=None):
    '''
    Load average ABR waveforms

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the average waveforms file itself.
    frequency : {None, float}
        If provided, return only this frequency.
    level : {None, float}
        If provided, return only this level.

    Returns
    -------
    df : pd.DataFrame
        Waveforms indexed by frequency and level, one column per timepoint.
        Pass through `cftsdata.util.to_long` for long format.
    '''
    df = pd.read_csv(filename, header=[0, 1, 2, 3], index_col=0)
    df = df.droplevel(['epoch_n', 'epoch_reject_ratio'], axis='columns')
    f = df.columns.get_level_values(0)
    # Convert frequencies to float if tones or Enum value for click
    f = [ABRStim.CLICK.value if v == 'click' else float(v) for v in f]
    l = df.columns.get_level_values(1).astype('float')
    df.columns = [f, l]
    df.columns.names = ['frequency', 'level']
    df = df.T
    try:
        if frequency is not None:
            df = df.xs(frequency, level='frequency', drop_level=False)
        if level is not None:
            df = df.xs(level, level='level', drop_level=False)
    except KeyError:
        # Not every dataset is run at every frequency and level. Return an
        # empty frame rather than raising so the result can still be
        # concatenated with the datasets that do have the requested stimulus.
        return pd.DataFrame()
    return df


@ensure_suffix(SUFFIXES['waveforms'])
def io(filename, level=None, rater=None, subthreshold_handling=None):
    '''
    Load the ABR peak analyses (i.e., input-output functions) for a dataset

    All frequencies and raters found in the dataset folder are loaded.

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the average waveforms file itself.
    level : {None, float}
        If provided, return only this level.
    rater : {None, str}
        If provided, return only the analyses by this rater.
    subthreshold_handling : {None, 'discard', 'zero', 'nan', 'na'}
        How to handle sub-threshold data. See `cftsdata.abr.load_abr_analysis`.

    Returns
    -------
    df : pd.DataFrame
        Peak measures indexed by frequency, threshold, rater and level.
    '''
    result = {}
    for m in sorted(filename.parent.glob(P_ANALYZED)):
        freq, file_th, file_rater, peaks = \
            load_abr_analysis(m, subthreshold_handling=subthreshold_handling)
        if rater is not None and file_rater != rater:
            continue
        result[freq, file_th, file_rater] = peaks
    if not result:
        return pd.DataFrame()
    df = pd.concat(result, names=['frequency', 'threshold', 'rater'])
    if level is not None:
        df = df.xs(level, level='level', drop_level=False)
    return df


def th(filename, rater=None, subthreshold_handling=None):
    '''
    Load the ABR thresholds picked by the rater(s) for a dataset

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or a single `-analyzed.txt` file.
    rater : {None, str}
        If provided, return only the thresholds set by this rater.

    Returns
    -------
    df : pd.DataFrame
        One row per frequency and rater, with a `threshold` column.
    '''
    filename = Path(filename)
    if filename.is_dir():
        filenames = sorted(filename.glob(P_ANALYZED))
    else:
        filenames = [filename]
    result = []
    for m in filenames:
        freq, file_th, file_rater, _ = \
            load_abr_analysis(m, subthreshold_handling=subthreshold_handling)
        if rater is not None and file_rater != rater:
            continue
        result.append({'frequency': freq, 'threshold': file_th,
                       'rater': file_rater})
    return pd.DataFrame(result, columns=['frequency', 'threshold', 'rater'])


@ensure_suffix(SUFFIXES['abrpresto_th'])
def abrpresto_th(filename):
    '''
    Load the ABRpresto-estimated thresholds

    Returns
    -------
    th : pd.Series
        Thresholds indexed by frequency. Clicks are reported as
        `ABRStim.CLICK.value`.
    '''
    data = json.loads(filename.read_text())
    thresholds = {}
    for k, v in data.items():
        if k == 'click':
            freq = ABRStim.CLICK.value
        else:
            freq = float(k)
        thresholds[freq] = v['threshold']
    result = pd.Series(thresholds)
    result.index.name = 'frequency'
    result.name = 'threshold'
    return result


@ensure_suffix(SUFFIXES['waveforms'])
def frequencies(filename):
    '''
    Load the set of frequencies the dataset was run at

    Read from the header of the average waveforms file, so this is much faster
    than loading the waveforms themselves.

    Returns
    -------
    frequencies : list
        Sorted frequencies. Clicks are reported as `ABRStim.CLICK.value`.
    '''
    def _convert_freq(x):
        x = x.strip()
        if x == 'click':
            return float(ABRStim.CLICK.value)
        return float(x)

    with filename.open() as fh:
        return sorted(set(_convert_freq(v) for v in fh.readline().split(',')[1:]))


@ensure_suffix(SUFFIXES['settings'])
def settings(filename):
    '''
    Load the experiment settings used to acquire the dataset
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
