'''
Summarize ABR adaptation data

Each trial is a train of tone bursts (or clicks). The trains are extracted and
filtered as a whole, artifacts are rejected on the whole train (as they are
during acquisition) and the trains are then cut into one repeat per burst.
Each step is a separate function so that they can be used on their own, e.g.:

.. code-block:: python

    from cftsdata import abr_adaptation
    from cftsdata import summarize_abr_adaptation as s

    fh = abr_adaptation.load(filename)
    trains = s.get_trains(fh)
    trains, valid = s.reject_trains(trains, fh.get_setting('reject_threshold'))
    repeats = fh.cut_repeats(trains)
    repeat_mean = s.average(repeats)
    amplitude = s.repeat_amplitude(repeat_mean)
    s.plot_amplitude(amplitude)

'''
import logging
log = logging.getLogger(__name__)

import argparse
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psiaudio.plot import waterfall_plot

from psidata.manager import add_default_options, process_files

from . import abr_adaptation
from .abr import freq_to_label
from .loaders.abr_adaptation import SUFFIXES
from .summarize_abr import _get_epochs, eeg_spectrum, plot_eeg_spectrum


COLUMNS = ['frequency', 'level', 'polarity']
CONDITIONS = ['frequency', 'level']


expected_suffixes = [
    SUFFIXES['waveforms'],
    SUFFIXES['train_waveforms'],
    SUFFIXES['amplitude'],
    SUFFIXES['trials'],
    SUFFIXES['processing_settings'],
    SUFFIXES['settings'],
    SUFFIXES['eeg_spectrum'],
    SUFFIXES['eeg_rms'],
    'ABR adaptation waveforms.pdf',
    'ABR adaptation train.pdf',
    'ABR adaptation amplitude.pdf',
    'ABR adaptation eeg spectrum.pdf',
]


################################################################################
# Analysis
################################################################################
def get_trains(fh, offset=-1e-3, repeat_duration=10e-3,
               filter_settings='saved', downsample=None, cb=None):
    '''
    Extract every train, without rejecting artifacts

    Parameters
    ----------
    fh : ABRAdaptationFile
        Dataset to analyze.
    offset : float
        Start of the train epoch (and of each repeat cut from it), in seconds,
        re. burst onset.
    repeat_duration : float
        The train epochs are long enough to cut out repeats of this duration.
    filter_settings : {None, 'saved', dict}
        See `cftsdata.summarize_abr.process_file`.
    downsample : {None, int}
        Decimation factor. See `cftsdata.abr_adaptation.choose_downsample` for
        choosing one that keeps the repeats on whole samples.
    '''
    duration = fh.get_train_duration(repeat_duration)
    return _get_epochs(fh, offset, duration, filter_settings,
                       downsample=downsample, cb=cb)


def reject_trains(trains, reject_threshold):
    '''
    Reject trains that exceed the threshold anywhere in the train

    Returns
    -------
    trains : pd.DataFrame
        The trains that were kept.
    valid : pd.Series
        True for each train that was kept, indexed by train.
    '''
    valid = (np.abs(trains) < reject_threshold).all(axis=1)
    return trains.loc[valid], valid


def limit_trains(trains, n):
    '''
    Keep the first `n` trains of each condition (half of each polarity)
    '''
    n = int(np.floor(n / 2))
    return trains.groupby(COLUMNS, group_keys=False) \
        .apply(lambda x: x.iloc[:n])


def trial_info(trains, valid):
    '''
    Number of trains averaged and fraction rejected for each condition

    Parameters
    ----------
    trains : pd.DataFrame
        Trains that were averaged.
    valid : pd.Series
        Reject mask returned by `reject_trains`.
    '''
    return pd.DataFrame({
        'epoch_n': trains.groupby(CONDITIONS).size(),
        'epoch_reject_ratio': 1 - valid.groupby(CONDITIONS).mean(),
    })


def average(epochs):
    '''
    Average across trains

    Works for both trains and repeats. The two polarities are averaged
    separately and then together so that each polarity is weighted equally
    (cancelling the stimulus artifact) even if the number of trains differ.
    '''
    grouping = [n for n in epochs.index.names if n != 't0']
    mean = epochs.groupby(grouping).mean()
    if 'polarity' in grouping:
        grouping.remove('polarity')
        mean = mean.groupby(grouping).mean()
    return mean


def repeat_amplitude(repeat_mean, window=(1e-3, 6e-3), repeat_t0=None):
    '''
    Amplitude of each repeat within the analysis window

    Parameters
    ----------
    repeat_mean : pd.DataFrame
        Average waveform of each repeat as returned by `average`.
    window : (float, float)
        Start and end of the analysis window, in seconds re. burst onset.
    repeat_t0 : {None, array}
        If provided, start of each repeat re. the start of the train (see
        `ABRAdaptationFile.repeat_t0`), which is added as a column.

    Returns
    -------
    amplitude : pd.DataFrame
        Peak-to-peak (`ptp`) and RMS (`rms`) amplitude in the window, and both
        normalized to the first repeat (`ptp_norm`, `rms_norm`).
    '''
    lb, ub = window
    w = repeat_mean.loc[:, lb:ub]
    amplitude = pd.DataFrame({
        'ptp': w.max(axis=1) - w.min(axis=1),
        'rms': np.sqrt((w ** 2).mean(axis=1)),
    })
    if repeat_t0 is not None:
        repeats = amplitude.index.get_level_values('repeat')
        amplitude.insert(0, 'repeat_t0', np.asarray(repeat_t0)[repeats])
    norm = normalize_amplitude(amplitude[['ptp', 'rms']])
    return amplitude.join(norm.add_suffix('_norm'))


def normalize_amplitude(amplitude, repeat=0):
    '''
    Express amplitude relative to that of the given repeat
    '''
    reference = amplitude.xs(repeat, level='repeat')
    return amplitude / reference.reindex(
        amplitude.index.droplevel('repeat')).values


################################################################################
# Plotting
################################################################################
def _panel_title(key):
    frequency, level = key
    return f'{freq_to_label(frequency, "k")}, {level:.0f} dB SPL'


def plot_repeat_waveforms(repeat_mean, window=None):
    '''
    Waterfall plot of the average waveform of each repeat

    One panel per frequency and level, with the first repeat at the top.

    Parameters
    ----------
    repeat_mean : pd.DataFrame
        Average waveform of each repeat as returned by `average`.
    window : {None, (float, float)}
        If provided, shade the window used to measure amplitude.
    '''
    grouped = repeat_mean.groupby(CONDITIONS)
    n_panels = len(grouped)
    n_repeats = repeat_mean.index.get_level_values('repeat').nunique()
    figure, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 2 + 0.2*n_repeats),
                                squeeze=False)
    for ax, (key, data) in zip(axes[0], grouped):
        # waterfall_plot draws the last row at the top.
        data = data.sort_index(level='repeat', ascending=False)
        waterfall_plot(ax, data, 'repeat', x_transform=lambda x: x*1e3,
                       plotkw={'lw': 0.75}, label_fmt=lambda x: str(x + 1))
        if window is not None:
            ax.axvspan(window[0]*1e3, window[1]*1e3, color='0.9', zorder=-1)
        ax.set_title(_panel_title(key))
        ax.set_xlabel('Time re. burst onset (msec)')
    return figure


def plot_train_waveforms(train_mean, repeat_t0):
    '''
    Average waveform of the whole train

    One panel per frequency and level. The onset of each burst is marked.
    '''
    grouped = train_mean.groupby(CONDITIONS)
    n_panels = len(grouped)
    figure, axes = plt.subplots(n_panels, 1, figsize=(12, 3*n_panels),
                                squeeze=False)
    for ax, (key, data) in zip(axes[:, 0], grouped):
        t = data.columns.values * 1e3
        ax.plot(t, data.iloc[0].values, 'k-', lw=0.5)
        for t0 in repeat_t0:
            ax.axvline(t0 * 1e3, color='0.8', lw=0.5, zorder=-1)
        ax.set_title(_panel_title(key))
        ax.set_xlabel('Time re. train onset (msec)')
        ax.set_ylabel('Amplitude (V)')
        ax.grid(False)
    return figure


def plot_amplitude(amplitude, measure='ptp'):
    '''
    Amplitude of each repeat as a function of its position in the train

    One panel per frequency, with one line per level. The top row shows the
    amplitude and the bottom row the amplitude normalized to the first repeat.

    Parameters
    ----------
    amplitude : pd.DataFrame
        Amplitude as returned by `repeat_amplitude`.
    measure : {'ptp', 'rms'}
        Amplitude measure to plot.
    '''
    grouped = amplitude.groupby('frequency')
    n_panels = len(grouped)
    figure, axes = plt.subplots(2, n_panels, figsize=(4*n_panels, 6),
                                squeeze=False, sharex=True)
    label = {'ptp': 'Peak-to-peak', 'rms': 'RMS'}[measure]
    for i, (frequency, data) in enumerate(grouped):
        for level, level_data in data.groupby('level'):
            repeat = level_data.index.get_level_values('repeat') + 1
            l, = axes[0, i].plot(repeat, level_data[measure], 'o-', ms=3,
                                 label=f'{level:.0f} dB SPL')
            axes[1, i].plot(repeat, level_data[f'{measure}_norm'], 'o-', ms=3,
                            color=l.get_color())
        axes[0, i].set_title(freq_to_label(frequency, 'k'))
        axes[1, i].axhline(1, color='k', ls=':', lw=0.5)
        axes[1, i].set_xlabel('Burst #')
    axes[0, 0].set_ylabel(f'{label} amplitude (V)')
    axes[1, 0].set_ylabel('Re. first burst')
    axes[0, -1].legend()
    return figure


################################################################################
# Summary
################################################################################
def process_file(filename, manager, offset=-1e-3, duration=10e-3,
                 filter_settings='saved', n_epochs='saved', target_fs=12.5e3,
                 amplitude_window=(1e-3, 6e-3), latency_correction=0,
                 gain_correction=1, eeg_duration=2.5):
    '''
    Extract ABR adaptation repeats, average and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR adaptation experiment.
    manager : instance of DatasetManager
        DatasetManager for handling data storage and review.
    offset : sec
        The start of each repeat, in seconds, re. burst onset. Negative values
        can be used to extract a prestimulus baseline (which, for all but the
        first burst, is the end of the previous repeat).
    duration : sec
        The duration of each repeat, in seconds.
    filter_settings : {None, 'saved', dict}
        If None, no additional filtering is done. If 'saved', uses the digital
        filter settings that were saved in the file. If a dictionary, must
        contain 'lb' (the lower bound of the passband in Hz), 'ub' (the upper
        bound of the passband in Hz) and 'order'. The whole train is filtered
        before it is cut into repeats.
    n_epochs : {None, 'saved', int}
        If None, all trains will be used. If 'saved', use the number of
        averages defined at acquisition time. If integer, will limit the
        number of trains per frequency and level to this number.
    target_fs : float
        Lowest acceptable sampling rate after decimation. The decimation
        factor is chosen so that the bursts stay on whole samples (see
        `cftsdata.abr_adaptation.choose_downsample`).
    amplitude_window : (float, float)
        Window, in seconds re. burst onset, in which to measure the amplitude
        of each repeat.
    latency_correction : float
        Correction, in seconds, to apply to timing of ABR. This allows us to
        retroactively correct for any ADC or DAC delays that were present in
        the acquisition system.
    gain_correction : float
        Correction to apply to the scaling of the waveform.
    eeg_duration : float
        Duration, in minutes, of EEG to analyze for RMS and PSD (useful for
        validating settings on amplifier).
    '''
    settings = locals()
    filename = Path(filename)
    settings.pop('manager')
    settings['filename'] = str(settings['filename'])
    settings['creation_time'] = dt.datetime.now().isoformat()

    fh = abr_adaptation.load(filename)
    if len(fh.erp_metadata) == 0:
        raise IOError('No data in file')

    md = fh.erp_metadata.iloc[:1].to_dict('records')[0]
    for column in COLUMNS:
        md.pop(column, None)
    del md['t0']

    downsample = abr_adaptation.choose_downsample(fh.eeg.fs, fh.repeat_period,
                                                  target_fs)
    settings['downsample'] = downsample
    settings['actual_fs'] = fh.eeg.fs / downsample
    settings['n_repeats'] = fh.n_repeats
    settings['repeat_period'] = fh.repeat_period

    if n_epochs == 'saved':
        n_epochs = fh.get_setting('averages')

    with manager.create_cb() as cb:
        def cb_rescale(frac):
            nonlocal cb
            cb(frac * 0.5)

        trains = get_trains(fh, offset + latency_correction, duration,
                            filter_settings, downsample=downsample,
                            cb=cb_rescale)
        if gain_correction != 1:
            trains = trains * gain_correction
        if latency_correction != 0:
            trains.columns = trains.columns - latency_correction

        reject_threshold = fh.get_setting('reject_threshold', reconcile='last')
        trains, valid = reject_trains(trains, reject_threshold)
        if n_epochs is not None:
            trains = limit_trains(trains, n_epochs)
        info = trial_info(trains, valid)
        cb(0.6)

        repeats = fh.cut_repeats(trains, offset, duration)
        repeat_mean = average(repeats)
        train_mean = average(trains)
        amplitude = repeat_amplitude(repeat_mean, amplitude_window,
                                     fh.repeat_t0)
        cb(0.7)

        manager.save_dict(settings, SUFFIXES['processing_settings'])
        manager.save_dict(md, SUFFIXES['settings'])
        manager.save_df(repeat_mean, SUFFIXES['waveforms'])
        manager.save_df(train_mean, SUFFIXES['train_waveforms'])
        manager.save_df(amplitude, SUFFIXES['amplitude'])
        manager.save_df(info, SUFFIXES['trials'])

        manager.save_fig(plot_repeat_waveforms(repeat_mean, amplitude_window),
                         'ABR adaptation waveforms.pdf')
        manager.save_fig(plot_train_waveforms(train_mean, fh.repeat_t0),
                         'ABR adaptation train.pdf')
        manager.save_fig(plot_amplitude(amplitude),
                         'ABR adaptation amplitude.pdf')
        cb(0.85)

        eeg_psd, eeg_rms = eeg_spectrum(fh, eeg_duration)
        manager.save_fig(plot_eeg_spectrum(eeg_psd),
                         'ABR adaptation eeg spectrum.pdf')
        manager.save_df(eeg_psd, SUFFIXES['eeg_spectrum'])
        manager.save_dict(eeg_rms, SUFFIXES['eeg_rms'])

        cb(1.0)
        plt.close('all')

    return True


def main():
    parser = argparse.ArgumentParser('Summarize ABR adaptation files in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*abr_adaptation*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
