'''
Loaders for the data products written by `cftsdata.summarize_mlr_llr`
'''
import pandas as pd

from ._util import ensure_suffix


SUFFIXES = {
    'abr_waveforms': 'ABR average waveforms.csv',
    'mlr_waveforms': 'MLR average waveforms.csv',
    'llr_waveforms': 'LLR average waveforms.csv',
}


def waveforms(filename):
    '''
    Load any of the average waveform files saved by
    `cftsdata.summarize_mlr_llr.process_file`

    The ABR, MLR and LLR files share a layout, so this reads whichever one it
    is given. Use `abr_waveforms`, `mlr_waveforms` or `llr_waveforms` to name
    the band instead, which also lets you pass the dataset folder rather than
    the file.

    Unlike `cftsdata.loaders.abr.waveforms`, there's no polarity/epoch_n/
    epoch_reject_ratio info in the header here -- `load_epochs` averages across
    polarity before saving, leaving a plain (frequency, level) row index and
    float-time columns.
    '''
    df = pd.read_csv(filename, index_col=['frequency', 'level'])
    df.columns = df.columns.astype(float)
    return df


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
