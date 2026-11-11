'''
Loaders for the data products written by `cftsdata.summarize_efr`
'''
import json

import numpy as np
import pandas as pd

from ._util import ensure_suffix


SUFFIXES = {
    'psd': 'EEG bootstrapped.csv',
    'harmonics': 'EFR harmonics.csv',
    # Written only by older versions of summarize_efr. The stimulus is now
    # checked by the separate summarize-efr-spl entrypoint (`mic_spectrum`).
    'level': 'stimulus levels.csv',
    'spl': 'stimulus SPL.csv',
    'processing_settings': 'EFR processing settings.json',
    # Written by `summarize_efr.process_file_spl`
    'mic_spectrum': 'mic spectrum.csv',
    'mic_levels': 'mic levels.csv',
    'mic_processing_settings': 'mic processing settings.json',
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


@ensure_suffix(SUFFIXES['psd'])
def psd(filename):
    return pd.read_csv(filename, index_col=['fm', 'fc', 'frequency'])


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
    result = df.groupby(grouping).mean()
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
        # Sum power (not amplitude) across the carrier and sidebands
        result = result.groupby(['fm', 'fc'])['level'] \
            .apply(lambda x: 10 * np.log10(np.sum(10 ** (x / 10)))) \
            .reset_index()
    return result


@ensure_suffix(SUFFIXES['mic_spectrum'])
def mic_spectrum(filename):
    '''
    Load the stimulus spectrum measured by each microphone channel

    Saved by `cftsdata.summarize_efr.process_file_spl`. Epochs are averaged
    within polarity, then combined across polarity into `component`:

    * 'difference', (pos - neg) / 2: the stimulus plus odd-order distortion.
    * 'sum', (pos + neg) / 2: even-order distortion, including any energy at
      the modulation frequency and its harmonics, which would survive the
      polarity averaging of the EEG.
    * 'single': the average for recordings with only one polarity.

    Returns
    -------
    df : pd.DataFrame
        Level (dB SPL) indexed by channel, fm, fc and component, one column per
        spectral frequency (Hz). NaN at frequencies outside the range of the
        microphone calibration.
    '''
    df = pd.read_csv(filename, index_col=['channel', 'fm', 'fc', 'component'])
    df.columns = df.columns.astype(float)
    return df


@ensure_suffix(SUFFIXES['mic_levels'])
def mic_levels(filename, sum_sidebands=True):
    '''
    Load the requested and actual stimulus level measured by each microphone

    Saved by `cftsdata.summarize_efr.process_file_spl`. Measured from the
    stimulus component (see `mic_spectrum`) at the carrier and sidebands,
    `fc + sideband * fm` (-1 to 1 for SAM, -10 to 10 for RAM).

    Parameters
    ----------
    filename : {str, pathlib.Path}
        Dataset folder or the mic levels file itself.
    sum_sidebands : bool
        If True, sum power across the carrier and sidebands to give the actual
        level of each stimulus.

    Returns
    -------
    df : pd.DataFrame
        If `sum_sidebands`, one row per channel, fm and fc with
        `requested_level`, `actual_level` and `difference` (actual - requested).
        Otherwise, one row per sideband with its `frequency` and `SPL`.
    '''
    df = pd.read_csv(filename)
    if not sum_sidebands:
        return df
    result = df.groupby(['channel', 'fm', 'fc']).agg(
        requested_level=('requested_level', 'first'),
        actual_level=('SPL', lambda x: 10 * np.log10(np.sum(10 ** (x / 10)))),
    )
    result['difference'] = result['actual_level'] - result['requested_level']
    return result.reset_index()


@ensure_suffix(SUFFIXES['mic_processing_settings'])
def mic_processing_settings(filename):
    '''
    Load the settings used by `cftsdata.summarize_efr.process_file_spl`

    Returns
    -------
    settings : dict
    '''
    return json.loads(filename.read_text())


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
