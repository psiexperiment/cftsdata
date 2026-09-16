'''
Loaders for the data products written by `cftsdata.summarize_memr` and
`cftsdata.summarize_memr_th`

The interleaved, simultaneous and sweep variants of the MEMR experiment all
save their results under the same names, so the same loaders serve all three.
'''
import json

import pandas as pd

from ._util import ensure_suffix


# Keyed by the name of the loader that reads each product; entries with no
# loader are here so that summarize_memr can build its expected-suffix lists
# from the same table.
SUFFIXES = {
    'psd': 'MEMR.csv',
    'psd_total': 'MEMR_total.csv',
    'amplitude': 'MEMR_amplitude.csv',
    'amplitude_total': 'MEMR_amplitude_total.csv',
    'elicitor_level': 'elicitor level.csv',
    'valid_count': 'valid count.csv',
    'processing_settings': 'processing settings.json',
}

# Written by summarize_memr_th, which prefixes each product with the method
# used to estimate threshold. Use `th_suffix` to build these.
TH_SUFFIXES = {
    'threshold': 'threshold.json',
    'raw_amplitude': 'raw amplitude.csv',
    'fitted_amplitude': 'fitted amplitude.csv',
    'two_sample': '2samp.csv',
    'processing_settings': 'processing settings.json',
}


def th_suffix(product, method='HT2'):
    '''
    Suffix for a threshold product saved by `summarize_memr_th`

    Parameters
    ----------
    product : str
        One of the keys of `TH_SUFFIXES`.
    method : str
        Method used to estimate threshold.
    '''
    try:
        return f'{method} {TH_SUFFIXES[product]}'
    except KeyError:
        products = ', '.join(TH_SUFFIXES)
        raise ValueError(f'Unrecognized product "{product}". '
                         f'Valid products are {products}.')


def _psd(filename, repeat=None):
    cols = ['repeat', 'elicitor_level', 'frequency']
    df = pd.read_csv(filename, index_col=cols)['amplitude']
    if repeat is not None:
        df = df.loc[repeat]
    return df


@ensure_suffix(SUFFIXES['psd'])
def psd(filename, repeat=None):
    '''
    Load the MEMR as a function of probe frequency

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the MEMR file itself.
    repeat : {None, int}
        If provided, return only this repeat.

    Returns
    -------
    amplitude : pd.Series
        Change in probe level (dB), indexed by repeat, elicitor level and
        probe frequency.
    '''
    return _psd(filename, repeat)


@ensure_suffix(SUFFIXES['psd_total'])
def psd_total(filename, repeat=None):
    '''
    Load the MEMR as a function of probe frequency, summed across repeats

    See `psd` for parameters.
    '''
    return _psd(filename, repeat)


def _amplitude(filename, repeat=None, span=None):
    cols = ['repeat', 'elicitor_level', 'span']
    df = pd.read_csv(filename, index_col=cols)['amplitude']
    if repeat is not None:
        df = df.loc[repeat]
    if span is not None:
        df = df.xs(span, level='span')
    return df.reset_index()


@ensure_suffix(SUFFIXES['amplitude'])
def amplitude(filename, repeat=None, span=None):
    '''
    Load the MEMR amplitude summarized over each frequency span

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the MEMR amplitude file itself.
    repeat : {None, int}
        If provided, return only this repeat.
    span : {None, str}
        If provided, return only this span (e.g., `'conventional'`).

    Returns
    -------
    df : pd.DataFrame
        One row per repeat, elicitor level and span.
    '''
    return _amplitude(filename, repeat, span)


@ensure_suffix(SUFFIXES['amplitude_total'])
def amplitude_total(filename, repeat=None, span=None):
    '''
    Load the MEMR amplitude per frequency span, summed across repeats

    See `amplitude` for parameters.
    '''
    return _amplitude(filename, repeat, span)


@ensure_suffix(SUFFIXES['elicitor_level'])
def elicitor_level(filename):
    '''
    Load the measured level of the MEMR elicitor
    '''
    return pd.read_csv(filename)


@ensure_suffix(th_suffix('threshold'))
def metrics(filename):
    '''
    Load the MEMR threshold and the metrics used to estimate it

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the threshold file itself. For a method other than
        the default, pass `suffix=memr.th_suffix('threshold', method)`.
    '''
    return pd.Series(json.loads(filename.read_text()))


@ensure_suffix(th_suffix('raw_amplitude'))
def io(filename):
    '''
    Load the amplitude used to estimate MEMR threshold

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the amplitude file itself. Pass
        `suffix=memr.th_suffix('fitted_amplitude')` for the fitted amplitude,
        or build a suffix for a different method the same way.
    '''
    return pd.read_csv(filename)
