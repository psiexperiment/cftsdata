import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from . import abr
from psidata.manager import add_default_options, process_files
from psiaudio import plot, util

import cftsdata
import psiaudio
import psidata

from .loaders.mlr_llr import SUFFIXES


EXPECTED_SUFFIXES = [
    SUFFIXES['abr_waveforms'],
    SUFFIXES['mlr_waveforms'],
    SUFFIXES['llr_waveforms'],
    SUFFIXES['abr_noise_power'],
    SUFFIXES['mlr_noise_power'],
    SUFFIXES['llr_noise_power'],
    'waveforms.pdf',
    SUFFIXES['processing_settings'],
]


# Epochs are grouped and averaged by these columns, so they are not
# configurable.
EPOCH_COLUMNS = ['frequency', 'level', 'polarity']

# Each band is extracted from the averaged epoch with a zero-phase
# (`filtfilt`) Butterworth bandpass, then trimmed to `window` (s).
FILTERS = {
    'ABR': {'order': 2, 'band': [300, 3000], 'window': [-0.001, 0.01]},
    'MLR': {'order': 2, 'band': [10, 100], 'window': [-0.01, 0.05]},
    'LLR': {'order': 2, 'band': [1, 30], 'window': [-0.05, 0.3]},
}

# Artifact reject is applied to the epochs filtered with the ABR band since the
# low frequencies in the MLR/LLR bands are dominated by slow drift. Epochs that
# pass are kept, unfiltered, for the averages.
REJECT_FILTER = {k: FILTERS['ABR'][k] for k in ('order', 'band')}

# Settings used by `process_file`. These are passed to `load_epochs` and saved
# to the processing settings file, so the file always matches what was run.
EPOCH_SETTINGS = {
    'average': True,
    'reject_threshold': None,
    'reject_mode': 'absolute value',
    'reject_filter': REJECT_FILTER,
    'averages': None,
    'offset': -100e-3,
    'duration': 400e-3,
    'target_fs': 10e3,
    'filters': FILTERS,
}


def get_downsample(fs, target_fs):
    return int(np.ceil(fs / target_fs))


def resolve_saved(fh, name, value):
    '''
    Return the value actually used for a setting that may be `'saved'`

    Mirrors how `ABRFile.get_epochs` looks up `'saved'` and `'saved-{reconcile}'`
    so that the processing settings record the number used, not just the
    request.
    '''
    if isinstance(value, str) and value.startswith('saved'):
        reconcile = value.split('-', 1)[1] if '-' in value else None
        return fh.get_setting(name, reconcile=reconcile)
    return value


def resolve_reject_mode(fh, reject_threshold, reject_mode):
    '''
    Return the reject mode `ABRFile._apply_reject` uses for `reject_threshold`
    '''
    if reject_threshold is None:
        return None
    if isinstance(reject_threshold, str) and reject_threshold.startswith('saved'):
        return fh.get_setting_default('reject_mode', 'absolute value')
    return reject_mode


def bandpass(epochs, order, band):
    '''
    Zero-phase Butterworth bandpass of each epoch (row)
    '''
    b, a = signal.butter(order, band, 'band', fs=epochs.attrs['fs'])
    result = pd.DataFrame(signal.filtfilt(b, a, epochs.values, axis=1),
                          index=epochs.index, columns=epochs.columns)
    result.attrs.update(epochs.attrs)
    return result


def noise_power(epochs, grouping):
    '''
    Noise power of the average of `epochs` at each time point

    This is the squared standard error of the mean (SEM²) of each group, from
    the trial-to-trial variance pooled within polarity, so a difference between
    the rarefaction and condensation responses (which the polarity average
    keeps) isn't counted as noise. It is the expected power of the average when
    there is no response, and assumes trials are independent.

    Subtract its mean over a window from the mean of the squared average over
    the same window to get the response power in that window with the noise
    removed (zero on average when there is no response).

    Parameters
    ----------
    epochs : pd.DataFrame
        Single trials, with `grouping` and `polarity` in the index.
    grouping : list of str
        Index levels to average over (e.g., frequency and level).

    Returns
    -------
    noise : pd.DataFrame
        SEM² (V²) indexed by `grouping`, one column per time point.
    '''
    by_polarity = grouping + ['polarity']
    deviation = epochs - epochs.groupby(by_polarity).transform('mean')
    sum_squares = (deviation ** 2).groupby(grouping).sum()
    n = epochs.groupby(grouping).size()
    n_polarity = epochs.groupby(by_polarity).size().groupby(grouping).size()
    variance = sum_squares.div(n - n_polarity, axis=0)
    return variance.div(n, axis=0)


def run_info(filename):
    '''
    Information about the processing run common to all settings files
    '''
    return {
        'filename': str(filename),
        'creation_time': dt.datetime.now().isoformat(),
        'version': {
            'cftsdata': cftsdata.__version__,
            'psiaudio': psiaudio.__version__,
            'psidata': getattr(psidata, '__version__', None),
        },
    }


def load_epochs(fh, average=True, reject_threshold=None,
                reject_mode='absolute value', reject_filter=None,
                averages=None, offset=-100e-3, duration=400e-3, target_fs=10e3,
                filters=None, return_noise=False):
    '''
    Load MLR/LLR epochs and extract the ABR, MLR and LLR bands

    Artifact reject (`reject_threshold`, `reject_mode`; see
    `ABRFile.get_epochs`) is evaluated over the full epoch after filtering with
    `reject_filter` (default `REJECT_FILTER`, the ABR band). The number of
    `averages` is then drawn from the epochs that pass.

    If `average` is True, `epoch_n` (number of epochs averaged) and
    `epoch_reject_ratio` (fraction of epochs rejected) are appended to the
    (frequency, level) index, as in the ABR average waveforms.

    If `return_noise` is True (requires `average`), also return the noise
    power of each band's average (see `noise_power`), trimmed to the band's
    window and with the same index as the averages. Each band is then filtered
    once, trial by trial, and both the band's average and its noise power come
    from those filtered trials. Without `return_noise`, the average is filtered
    instead, which is cheaper. Filtering is linear, so the two averages agree
    up to rounding.

    Returns
    -------
    result : dict
        Waveforms for `raw` and each band in `filters`.
    noise : dict
        Only if `return_noise`. Noise power (V²) for each band in `filters`.
    '''
    if return_noise and not average:
        raise ValueError('return_noise requires average')
    if filters is None:
        filters = FILTERS
    if reject_filter is None:
        reject_filter = REJECT_FILTER
    downsample = get_downsample(fh.eeg.fs, target_fs)
    # Reject and averages are applied below so the reject can be done on the
    # filtered signal.
    epochs = fh.get_epochs(offset=offset, duration=duration,
                           columns=EPOCH_COLUMNS, reject_threshold=None,
                           averages=None, downsample=downsample)

    grouping = ['frequency', 'level']
    if reject_threshold is not None:
        filtered = bandpass(epochs, **reject_filter)
        keep = fh._apply_reject(filtered, reject_threshold, reject_mode).index
        accepted = pd.Series(epochs.index.isin(keep), index=epochs.index)
        epochs = epochs.loc[accepted.values]
    else:
        accepted = pd.Series(True, index=epochs.index)
    epoch_reject_ratio = 1 - accepted.groupby(grouping).mean()
    epochs = fh._apply_n(epochs, averages)

    trials = epochs
    if average:
        epoch_n = epochs.groupby(grouping).size()
        epochs = epochs.groupby(grouping).mean()
        epoch_info = pd.DataFrame({
            'epoch_n': epoch_n,
            'epoch_reject_ratio': epoch_reject_ratio,
        }).loc[epochs.index]
        info_index = pd.MultiIndex.from_frame(epoch_info.reset_index())
        epochs.index = info_index

    result = {'raw': epochs}
    noise = {}
    for name, f in filters.items():
        lb, ub = f['window']
        if return_noise:
            # Average and noise power from the same filtered trials.
            filtered = bandpass(trials, f['order'], f['band']).loc[:, lb:ub]
            band_average = filtered.groupby(grouping).mean().loc[epoch_info.index]
            band_noise = noise_power(filtered, grouping).loc[epoch_info.index]
            band_average.index = info_index
            band_noise.index = info_index
            result[name] = band_average
            noise[name] = band_noise
        else:
            # Filtering the average is much cheaper than filtering every trial
            # and gives the same result, since filtering is linear.
            result[name] = bandpass(epochs, f['order'], f['band']).loc[:, lb:ub]
    if return_noise:
        return result, noise
    return result


def plot_epochs(epochs):
    freq = epochs['ABR'].index.unique('frequency')
    figure, axes = plt.subplots(3, len(freq), figsize=(4 * len(freq), 12), squeeze=False)

    epochs_grouped = {k: v.groupby('frequency') for k, v in epochs.items()}
    for col, f in zip(axes.T, freq):
        for ax, name in zip(col, ('ABR', 'MLR', 'LLR')):
            ax.set_xlabel('Time (s)')
            w = epochs_grouped[name].get_group(f)
            plot.waterfall_plot(ax, w)
            if name == 'MLR':
                ax.axvspan(0, 0.01, alpha=0.25)
                ax.text(0, 1, 'ABR', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=8)
            elif name == 'LLR':
                ax.axvspan(0, 0.05, alpha=0.25)
                ax.text(0, 1, 'MLR', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=8)
        col[0].set_title(str(f))
    axes[0, 0].set_ylabel('ABR')
    axes[1, 0].set_ylabel('MLR')
    axes[2, 0].set_ylabel('LLR')
    return figure


def process_file(filename, manager):
    with manager.create_cb() as cb:
        fh = abr.load(filename)
        epochs, noise = load_epochs(fh, return_noise=True, **EPOCH_SETTINGS)
        figure = plot_epochs(epochs)

        settings = {
            **run_info(filename),
            **EPOCH_SETTINGS,
            'columns': EPOCH_COLUMNS,
            'reject_threshold_value': resolve_saved(
                fh, 'reject_threshold', EPOCH_SETTINGS['reject_threshold']),
            'reject_mode_value': resolve_reject_mode(
                fh, EPOCH_SETTINGS['reject_threshold'],
                EPOCH_SETTINGS['reject_mode']),
            'averages_value': resolve_saved(
                fh, 'averages', EPOCH_SETTINGS['averages']),
            'eeg_fs': fh.eeg.fs,
            'downsample': get_downsample(fh.eeg.fs, EPOCH_SETTINGS['target_fs']),
            'epoch_fs': epochs['raw'].attrs['fs'],
            'filter_type': 'butterworth bandpass, zero-phase (filtfilt)',
            'noise_power': 'squared SEM of the average: within-polarity '
                           'variance of the filtered trials, pooled, / epoch_n',
        }

        for k, v in epochs.items():
            manager.save_dataframe(v, f'{k} average waveforms.csv')
        for k, v in noise.items():
            manager.save_dataframe(v, SUFFIXES[f'{k.lower()}_noise_power'])
        manager.save_fig(figure, 'waveforms.pdf')
        manager.save_dict(settings, SUFFIXES['processing_settings'])


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize MLR/LLR files in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*mlr_llr_io*', fn=process_file,
                  expected_suffixes=EXPECTED_SUFFIXES, **args)


# Separate entrypoint for processing SPL

EXPECTED_SUFFIXES_SPL = [
    SUFFIXES['stimulus_waveforms'],
    SUFFIXES['stimulus_spl'],
    'mic.pdf',
    SUFFIXES['stimulus_processing_settings'],
]


# Settings used by `process_file_spl`. Artifact reject and averages settings
# saved with the recording are for the EEG and do not apply to the microphone.
MIC_SETTINGS = {
    'signal': 'microphone_monitor_monitor',
    'offset': 0,
    'duration': 20e-3,
    'reject_threshold': None,
    'averages': None,
}


def plot_spl(mic_mean, spl):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for (f, l), df in mic_mean.groupby(['frequency', 'level']):
        label = f'{f}, {l}'
        df = df.droplevel(['frequency', 'level'])
        # Skip a polarity if it wasn't presented
        if -1 in df.index:
            axes[0].plot(df.loc[-1], label=label)
        if 1 in df.index:
            axes[1].plot(df.loc[1], label=label)
        axes[2].plot(spl.loc[f, l], label=label)
    axes[0].set_title('Rarefaction')
    axes[1].set_title('Condensation')
    for ax in axes[:2]:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (V)')
    axes[2].set_title('Spectrum')
    axes[2].set_xscale('octave')
    axes[2].axis(xmin=250, xmax=spl.columns.max())
    axes[2].set_xlabel('Frequency (kHz)')
    axes[2].set_ylabel('Level (dB SPL)')
    axes[2].legend(bbox_to_anchor=(1, 1), loc='upper left')
    return figure


def process_file_spl(filename, manager):
    with manager.create_cb() as cb:
        fh = abr.load(filename)
        mic_signal = getattr(fh, MIC_SETTINGS['signal'])
        fs = mic_signal.fs
        mic = fh.get_epochs(columns=EPOCH_COLUMNS, cb=cb, **MIC_SETTINGS)
        mic_mean = mic.groupby(EPOCH_COLUMNS).mean()
        cal = mic_signal.get_calibration()
        # Drop the DC bin (can't be shown on an octave axis).
        spl_polarity = cal.get_db(util.psd_df(mic_mean, fs)).iloc[:, 1:]
        # Average power (not waveforms, which would cancel) across polarity.
        spl = spl_polarity.groupby(['frequency', 'level']) \
            .apply(lambda x: 10 * np.log10((10 ** (x / 10)).mean()))

        settings = {
            **run_info(filename),
            **MIC_SETTINGS,
            'columns': EPOCH_COLUMNS,
            'detrend': 'constant',
            'mic_fs': fs,
            'calibration': type(cal).__name__,
            'psd_window': None,
            'drop_dc': True,
            'polarity_average': 'power',
            'polarities': sorted(mic_mean.index.unique('polarity').tolist()),
        }

        figure = plot_spl(mic_mean, spl)
        manager.save_df(mic_mean, SUFFIXES['stimulus_waveforms'])
        manager.save_df(spl, SUFFIXES['stimulus_spl'])
        manager.save_fig(figure, 'mic.pdf')
        manager.save_dict(settings, SUFFIXES['stimulus_processing_settings'])


def main_spl():
    import argparse
    parser = argparse.ArgumentParser('Summarize MLR/LLR monitor microphone in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*mlr_llr_io*', fn=process_file_spl,
                  expected_suffixes=EXPECTED_SUFFIXES_SPL, **args)
