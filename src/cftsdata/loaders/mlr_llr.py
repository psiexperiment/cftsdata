'''
Loaders for the data products written by `cftsdata.summarize_mlr_llr`
'''
import json
from pathlib import Path

import pandas as pd

from ._util import ensure_suffix


SUFFIXES = {
    'abr_waveforms': 'ABR average waveforms.csv',
    'mlr_waveforms': 'MLR average waveforms.csv',
    'llr_waveforms': 'LLR average waveforms.csv',
    'stimulus_waveforms': 'stimulus waveforms.csv',
    'stimulus_spl': 'stimulus SPL.csv',
    'processing_settings': 'MLR LLR processing settings.json',
    'stimulus_processing_settings': 'stimulus processing settings.json',
}


def waveforms(filename):
    '''
    Load any of the average waveform files saved by
    `cftsdata.summarize_mlr_llr.process_file`

    The ABR, MLR and LLR files share a layout, so this reads whichever one it
    is given. Use `abr_waveforms`, `mlr_waveforms` or `llr_waveforms` to name
    the band instead, which also lets you pass the dataset folder rather than
    the file.

    `load_epochs` averages across polarity before saving, leaving one row per
    frequency and level and float-time columns. As in
    `cftsdata.loaders.abr.waveforms`, the `epoch_n` and `epoch_reject_ratio`
    index levels are dropped; use `epoch_info` to load them.
    '''
    df = _read_waveforms(filename)
    return df.droplevel([n for n in EPOCH_INFO if n in df.index.names])


# Saved in the index of the average waveform files. Files processed before
# these were added have only frequency and level.
EPOCH_INFO = ['epoch_n', 'epoch_reject_ratio']


def _read_waveforms(filename):
    df = pd.read_csv(filename)
    index = [c for c in ['frequency', 'level'] + EPOCH_INFO if c in df]
    df = df.set_index(index)
    df.columns = df.columns.astype(float)
    return df


def epoch_info(filename):
    '''
    Load the number of epochs averaged and the fraction rejected

    These are the same in the ABR, MLR and LLR average waveform files, so any
    of them can be passed. Given the dataset folder, the ABR file is read.

    Returns
    -------
    df : pd.DataFrame
        `epoch_n` and `epoch_reject_ratio`, indexed by frequency and level.
        Empty columns if the dataset was processed before these were saved.
    '''
    filename = Path(filename)
    waveform_suffixes = [SUFFIXES[f'{b}_waveforms'] for b in ('abr', 'mlr', 'llr')]
    if not any(filename.name.endswith(s) for s in waveform_suffixes):
        filename = filename / f'{filename.stem} {SUFFIXES["abr_waveforms"]}'
    index = _read_waveforms(filename).index.to_frame(index=False)
    return index.reindex(columns=['frequency', 'level'] + EPOCH_INFO) \
        .set_index(['frequency', 'level'])


@ensure_suffix(SUFFIXES['abr_waveforms'])
def abr_waveforms(filename):
    '''
    Load the average ABR waveforms recorded during an MLR/LLR experiment

    Returns
    -------
    df : pd.DataFrame
        Waveforms indexed by frequency and level, one column per timepoint.
        Pass through `cftsdata.util.to_long` for long format.
    '''
    return waveforms(filename)


@ensure_suffix(SUFFIXES['mlr_waveforms'])
def mlr_waveforms(filename):
    '''
    Load the average MLR waveforms

    See `abr_waveforms` for the return value.
    '''
    return waveforms(filename)


@ensure_suffix(SUFFIXES['llr_waveforms'])
def llr_waveforms(filename):
    '''
    Load the average LLR waveforms

    See `abr_waveforms` for the return value.
    '''
    return waveforms(filename)


@ensure_suffix(SUFFIXES['stimulus_waveforms'])
def stimulus_waveforms(filename):
    '''
    Load the average stimulus waveforms recorded by the monitor microphone

    Saved by `cftsdata.summarize_mlr_llr.process_file_spl`.

    Returns
    -------
    df : pd.DataFrame
        Waveforms (V) indexed by frequency, level and polarity, one column per
        timepoint (s).
    '''
    df = pd.read_csv(filename, index_col=['frequency', 'level', 'polarity'])
    df.columns = df.columns.astype(float)
    return df


@ensure_suffix(SUFFIXES['stimulus_spl'])
def stimulus_spl(filename):
    '''
    Load the stimulus spectrum measured by the monitor microphone

    Saved by `cftsdata.summarize_mlr_llr.process_file_spl`. The spectrum is
    the power average of the rarefaction and condensation spectra.

    Returns
    -------
    df : pd.DataFrame
        Level (dB SPL) indexed by frequency and level, one column per
        spectral frequency (Hz).
    '''
    df = pd.read_csv(filename, index_col=['frequency', 'level'])
    df.columns = df.columns.astype(float)
    return df


@ensure_suffix(SUFFIXES['processing_settings'])
def processing_settings(filename):
    '''
    Load the settings used by `cftsdata.summarize_mlr_llr.process_file`

    Returns
    -------
    settings : dict
    '''
    return json.loads(filename.read_text())


@ensure_suffix(SUFFIXES['stimulus_processing_settings'])
def stimulus_processing_settings(filename):
    '''
    Load the settings used by `cftsdata.summarize_mlr_llr.process_file_spl`

    Returns
    -------
    settings : dict
    '''
    return json.loads(filename.read_text())
