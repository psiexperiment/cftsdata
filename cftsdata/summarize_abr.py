import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import argparse
import datetime as dt
from functools import partial
from math import ceil
import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from psiaudio.plot import waterfall_plot
from psiaudio import util

from . import abr

from .util import add_default_options, process_files


COLUMNS = ['frequency', 'level', 'polarity']


abr_expected_suffixes = [
    'ABR average waveforms.csv',
    'ABR processing settings.json',
    'ABR experiment settings.json',
    'ABR waveforms.pdf',
    'ABR eeg spectrum.pdf',
    'ABR eeg spectrum.csv',
    'ABR eeg rms.json',
]


def load_abr_waveforms(filename):
	df = pd.read_csv(filename, header=[0, 1, 2, 3], index_col=0)
	df = df.droplevel(['epoch_n', 'epoch_reject_ratio'], axis='columns')
	f = df.columns.get_level_values(0).astype('float')
	l = df.columns.get_level_values(1).astype('float')
	df.columns = [f, l]
	return df.T


def _get_filter(fh):
    if not isinstance(fh, (abr.ABRFile, abr.ABRSupersetFile)):
        fh = abr.load(fh)
    return {
        'digital_filter': fh.get_setting_default('digital_filter', True),
        'lb': fh.get_setting_default('digital_highpass', 300),
        'ub': fh.get_setting_default('digital_lowpass', 3000),
        # Filter order is not currently an option in the psiexperiment ABR
        # program so it defaults to 1.
        'order': 1,
    }


def _get_epochs(fh, offset, duration, filter_settings, reject_ratio=None,
                downsample=None, cb=None):
    # We need to do the rejects in this code so that we can obtain the
    # information for generating the CSV files. Set both reject_threshold and
    # averages to None to ensure that all acquired trials are returned.
    kwargs = {'offset': offset, 'duration': duration, 'columns': COLUMNS,
              'reject_threshold': None, 'averages': None, 'downsample':
              downsample, 'cb': cb}

    if filter_settings is None:
        return fh.get_epochs(**kwargs)

    if filter_settings == 'saved':
        settings = _get_filter(fh)
        if not settings['digital_filter']:
            return fh.get_epochs(**kwargs)
        lb = settings['lb']
        ub = settings['ub']
        order = settings['order']
        kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
        return fh.get_epochs_filtered(**kwargs)

    lb = filter_settings['lb']
    ub = filter_settings['ub']
    order = filter_settings['order']
    kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
    return fh.get_epochs_filtered(**kwargs)


def add_trial(epochs):
    '''
    This adds trial number on a per-stim-level/frequency basis
    '''
    def number_trials(subset):
        subset = subset.sort_index(level='t0')
        idx = subset.index.to_frame()
        i = len(idx.columns) - 1
        idx.insert(i, 'trial', np.arange(len(idx)))
        subset.index = pd.MultiIndex.from_frame(idx)
        return subset

    levels = list(epochs.index.names[:-1])
    if 'polarity' in levels:
        levels.remove('polarity')

    return epochs.groupby(levels, group_keys=False).apply(number_trials)


def plot_waveforms_cb(epochs_mean, filename, name):
    epochs_mean = epochs_mean.reset_index(['epoch_n', 'epoch_reject_ratio'], drop=True)
    grouped = epochs_mean.groupby('frequency')
    n_panels = len(grouped)
    # Normally we want a 1D array of axes, but we have to disable squeezing
    # because if n_panels is 1, then we end up with an axes instead of a 1D
    # array of axes. So, we get a 2D array and then extract the first row.
    figure, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 8.5), squeeze=False)
    for ax, (frequency, data) in zip(axes[0], grouped):
        waterfall_plot(ax, data)
        ax.set_xlabel('Time (msec)')
        ax.set_title(f'{frequency * 1e-3:0.2f} Hz')
    figure.suptitle(name)
    figure.savefig(filename)


def process_file(filename, manager, offset=-1e-3, duration=10e-3,
                 filter_settings='saved', n_epochs='saved',
                 target_fs=12.5e3, analysis_window=None, latency_correction=0,
                 gain_correction=1, debug_mode=False,
                 plot_waveforms_cb=plot_waveforms_cb, eeg_duration=2.5):
    '''
    Extract ABR epochs, filter and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR experiment. If it's a set of ABR experiments, epochs across
        all experiments will be combined for the analysis.
    manager : instance of DatasetManager
        DatasetManager for handling data storage and review.
    offset : sec
        The start of the epoch to extract, in seconds, relative to tone pip
        onset. Negative values can be used to extract a prestimulus baseline.
    duration: sec
        The duration of the epoch to extract, in seconds, relative to the
        offset. If offset is set to -0.001 sec and duration is set to 0.01 sec,
        then the epoch will be extracted from -0.001 to 0.009 sec re tone pip
        onset.
    filter_settings : {None, 'saved', dict}
        If None, no additional filtering is done. If 'saved', uses the digital
        filter settings that were saved in the ABR file. If a dictionary, must
        contain 'lb' (the lower bound of the passband in Hz) and 'ub' (the
        upper bound of the passband in Hz).
    n_epochs : {None, 'saved', int, dict}
        If None, all epochs will be used. If 'saved', use the value defined at
        acquisition time. If integer, will limit the number of epochs per
        frequency and level to this number.
    target_fs : float
        Closest sampling rate to target
    analysis_window : Ignored
        This is ignored for now. Primarily used to allow acceptance of the
        queue since we add analysis window for GUI purposes.
    latency_correction : float
        Correction, in seconds, to apply to timing of ABR. This allows us to
        retroactively correct for any ADC or DAC delays that were present in
        the acquisition system.
    gain_correction : float
        Correction to apply to the scaling of the waveform. This allows us to
        retroactively correct for differences in gain that were present in the
        acquisition system.
    debug_mode : bool
        This is reserved for internal use only. This mode will load the epochs
        and return them without saving to disk.
    plot_waveforms_cb : {Callable, None}
        Callback that takes three arguments. Epoch mean dataframe, path to file
        to save figures in, and name of file.
    eeg_duration : float
        Duration, in seconds, of EEG to analyze for RMS and PSD (useful for
        validating settings on amplifier).
    '''
    settings = locals()
    filename = Path(filename)

    # Cleanup settings so that it is JSON-serializable
    settings.pop('manager')
    settings.pop('plot_waveforms_cb')
    settings['filename'] = str(settings['filename'])
    settings['creation_time'] = dt.datetime.now().isoformat()

    fh = abr.load(filename)
    if len(fh.erp_metadata) == 0:
        raise IOError('No data in file')

    # This is a hack to ensure that native Python types are returned instead of
    # Numpy ones. Newer versions of Pandas have fixed this issue, though.
    md = fh.erp_metadata.iloc[:1].to_dict('records')[0]
    for column in COLUMNS:
        del md[column]
    del md['t0']

    downsample = int(ceil(fh.eeg.fs / target_fs))
    settings['downsample'] = downsample
    settings['actual_fs'] = fh.eeg.fs / downsample

    if n_epochs is not None:
        if n_epochs == 'saved':
            n_epochs = fh.get_setting('averages')

    # Load the epochs. The callbacks for loading the epochs return a value in
    # the range 0 ... 1. Since this only represents "half" the total work we
    # need to do, rescale to the range 0 ... 0.5.
    with manager.create_cb() as cb:
        def cb_rescale(frac):
            nonlocal cb
            cb(frac * 0.5)

        epochs = _get_epochs(fh, offset + latency_correction, duration,
                            filter_settings, cb=cb_rescale, downsample=downsample)

        if gain_correction != 1:
            epochs = epochs * gain_correction

        if latency_correction != 0:
            new_idx = [(*r[:-1], r[-1] - latency_correction) for r in epochs.index]
            new_idx = pd.MultiIndex.from_tuples(new_idx, names=epochs.index.names)
            new_col = epochs.columns - latency_correction
            epochs = pd.DataFrame(epochs.values, index=new_idx, columns=new_col)

        if debug_mode:
            return epochs

        # Apply the reject
        reject_threshold = fh.get_setting('reject_threshold')
        m = np.abs(epochs) < reject_threshold
        m = m.all(axis=1)
        epochs = epochs.loc[m]

        cb(0.6)
        if n_epochs is not None:
            n = int(np.floor(n_epochs / 2))
            epochs = epochs.groupby(COLUMNS, group_keys=False) \
                .apply(lambda x: x.iloc[:n])
        cb(0.7)

        epoch_mean = epochs.groupby(COLUMNS).mean().groupby(COLUMNS[:-1]).mean()

        epoch_reject_ratio = 1-m.groupby(COLUMNS[:-1]).mean()
        epoch_n = epochs.groupby(COLUMNS[:-1]).size()
        epoch_info = pd.DataFrame({
            'epoch_n': epoch_n,
            'epoch_reject_ratio': epoch_reject_ratio,
        })
        if not np.all(epoch_mean.index == epoch_info.index):
            raise ValueError('Programming issue. Please contact developer.')

        # Merge in the N and reject ratio into the index for epoch_mean
        epoch_info = epoch_info.set_index(['epoch_n', 'epoch_reject_ratio'],
                                        append=True)
        epoch_mean.index = epoch_info.index
        epoch_mean.columns.name = 'time'

        manager.get_proc_filename('ABR processing settings.json') \
            .write_text(json.dumps(settings, indent=2))
        manager.get_proc_filename('ABR experiment settings.json') \
            .write_text(json.dumps(md, indent=2))

        epoch_mean.T.to_csv(manager.get_proc_filename('ABR average waveforms.csv'))
        cb(0.85)

        if plot_waveforms_cb is not None:
            plot_waveforms_cb(
                epoch_mean,
                manager.get_proc_filename('ABR waveforms.pdf'),
                filename.stem
            )

        # Get the first 2.5 minutes of the EEG. In general, even a
        # single-frequency ABR at 40/s will be a little over three minutes
        # long, so this should ensure we have a consistent duration baseline
        # against which we can compare all datasets.
        eeg = fh.eeg.get_segment(0, 0, eeg_duration*60, channel=0)
        n_averages = int(60 * eeg_duration)
        eeg_psd = util.db(util.psd_df(eeg, fs=fh.eeg.fs, waveform_averages=n_averages))

        eeg_figure, axes = plt.subplots(1, 1, figsize=(4, 4))
        axes.plot(eeg_psd.iloc[1:], 'k-')
        axes.set_xscale('octave', octaves=2)
        axes.axis(xmin=10, xmax=10e3)
        axes.set_xlabel('Frequency (kHz)')
        axes.set_ylabel('PSD (dB re 1V)')
        axes.axvline(60, ls=':', color='k')
        manager.save_fig(eeg_figure, 'ABR eeg spectrum.pdf')
        manager.save_dataframe(eeg_psd.rename('psd'), 'ABR eeg spectrum.csv')
        manager.save_dict({
            'eeg_rms': util.rms(eeg.values),
            'eeg_n_averages': n_averages,
        }, 'ABR eeg rms.json')

        cb(1.0)
        plt.close('all')

    return True


def main():
    parser = argparse.ArgumentParser('Filter and summarize ABR files in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    fn = partial(process_file, filter_settings='saved', offset=-0.001,
                 duration=0.01)
    process_files(glob_pattern='**/*abr_io*', fn=fn,
                  expected_suffixes=abr_expected_suffixes, **args)


def main_gui():
    import enaml
    from enaml.qt.qt_application import QtApplication
    with enaml.imports():
        from .summarize_abr_gui import SummarizeABRGui
    app = QtApplication()
    view = SummarizeABRGui()
    view.show()
    app.start()
