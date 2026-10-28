'''
Loaders for the data products written by `cftsdata.summarize_dpoae`
'''
import pandas as pd

from ._util import ensure_suffix


SUFFIXES = {
    'io': 'io.csv',
    'th': 'th.csv',
    'dpgram': 'dpgram.csv',
}


@ensure_suffix(SUFFIXES['io'])
def io(filename):
    '''
    Load the DPOAE input-output functions

    Returns
    -------
    df : pd.DataFrame
        One row per F2 frequency and level.
    '''
    return pd.read_csv(filename)


@ensure_suffix(SUFFIXES['th'])
def th(filename, criterion=None):
    '''
    Load the DPOAE thresholds

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the threshold file itself.
    criterion : {None, float}
        If provided, return only the threshold at this criterion.

    Returns
    -------
    th : pd.Series
        Thresholds indexed by F2 frequency and criterion.
    '''
    df = pd.read_csv(filename, index_col=0)
    df.columns = df.columns.astype('f')
    if criterion is not None:
        df = df.loc[:, [criterion]]
    df.columns.name = 'criterion'
    return df.stack().rename('threshold')


@ensure_suffix(SUFFIXES['dpgram'])
def dpgram(filename):
    '''
    Load the DPgram

    Returns
    -------
    df : pd.DataFrame
        One row per F2 frequency and level.
    '''
    return pd.read_csv(filename)
