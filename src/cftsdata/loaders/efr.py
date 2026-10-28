'''
Loaders for the data products written by `cftsdata.summarize_efr`
'''
import pandas as pd

from psiaudio import util

from ._util import ensure_suffix


SUFFIXES = {
    'eeg_bootstrapped': 'EEG bootstrapped.csv',
    'harmonics': 'EFR harmonics.csv',
    'level': 'stimulus levels.csv',
    'spl': 'stimulus SPL.csv',
    'processing_settings': 'EFR processing settings.json',
    # Not written by summarize_efr. The total power is saved alongside the
    # harmonics by the analysis itself.
    'power': 'EFR.csv',
}

# The Verhulst analysis saves its output under different names from the
# Bharadwaj analysis, and the modifier is not the same for the two products.
# Use `harmonics_suffix` and `power_suffix` rather than reading this directly.
METHOD_SUFFIXES = {
    'harmonics': {
        'bharadwaj': SUFFIXES['harmonics'],
        'verhulst': 'EFR harmonics linear.csv',
    },
    'power': {
        'bharadwaj': SUFFIXES['power'],
        'verhulst': 'EFR amplitude linear.csv',
    },
}


def _suffix(product, method):
    try:
        return METHOD_SUFFIXES[product][method.lower()]
    except KeyError:
        methods = ', '.join(METHOD_SUFFIXES[product])
        raise ValueError(f'Unrecognized method "{method}". '
                         f'Valid methods are {methods}.')


@ensure_suffix(SUFFIXES['harmonics'])
def harmonics(filename):
    '''
    Load the EFR response at each harmonic of the modulation frequency

    Averaged across bootstraps.

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the harmonics file itself. For the Verhulst
        analysis, pass `suffix=efr.harmonics_suffix('verhulst')`.

    Returns
    -------
    df : pd.DataFrame
        One row per carrier frequency, modulation frequency and harmonic.
    '''
    df = pd.read_csv(filename)
    # This corrects for a difference in the data saved between the Bharadwaj
    # and Verhulst bootstrapping approaches. In Bharadwaj output, the harmonic
    # is the harmonic number (i.e., not the actual harmonic frequency). In the
    # Verhulst output, we only save the frequency, not the harmonic number.
    if 'frequency' not in df:
        df['frequency'] = df['harmonic']
        df['harmonic'] = df.eval('frequency / fm').astype('i')
    grouping = ['fc', 'fm', 'frequency', 'harmonic']
    result = df.groupby(grouping).mean().reset_index()
    if 'bootstrap' in result:
        return result.drop(columns='bootstrap')
    return result


@ensure_suffix(SUFFIXES['power'])
def power(filename):
    '''
    Load the total EFR power

    Averaged across bootstraps.

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the EFR file itself. For the Verhulst analysis, pass
        `suffix=efr.power_suffix('verhulst')`.

    Returns
    -------
    df : pd.DataFrame
        One row per carrier and modulation frequency.
    '''
    result = pd.read_csv(filename) \
        .groupby(['fc', 'fm']) \
        .mean().reset_index()
    if 'bootstrap' in result:
        return result.drop(columns='bootstrap')
    return result


@ensure_suffix(SUFFIXES['level'])
def level(filename, sum_harmonics=True):
    '''
    Load the measured level of the EFR stimulus

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the stimulus levels file itself.
    sum_harmonics : bool
        If True, sum across harmonics to give the total level of each stimulus.

    Returns
    -------
    df : pd.DataFrame
        One row per carrier and modulation frequency, with a `level` column.
    '''
    result = pd.read_csv(filename).rename(columns={'SPL': 'level'})
    if sum_harmonics:
        result = result.groupby(['fm', 'fc'])['level'] \
            .apply(lambda x: util.db(util.dbi(x).sum())) \
            .reset_index()
    return result


def harmonics_suffix(method):
    '''
    Suffix for the harmonics file saved by the given analysis method

    Parameters
    ----------
    method : {'bharadwaj', 'verhulst'}
    '''
    return _suffix('harmonics', method)


def power_suffix(method):
    '''
    Suffix for the EFR power file saved by the given analysis method

    Parameters
    ----------
    method : {'bharadwaj', 'verhulst'}
    '''
    return _suffix('power', method)
