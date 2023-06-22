from pathlib import Path

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import palettable

from psiaudio.plot import iter_colors, waterfall_plot
from psiaudio import util

from .memr import InterleavedMEMRFile, SimultaneousMEMRFile
from .util import add_default_options, DatasetManager, process_files


expected_suffixes = [
    'stimulus train.pdf',
    'elicitor PSD.pdf',
    'probe PSD.pdf',
    'probe level.pdf',
    'MEMR.pdf',
]


def plot_stim_train(epochs, settings):
    figsize = 6, 1 * len(epochs)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    waterfall_plot(ax, epochs, 'elicitor_level', scale_method='max',
                   plotkw={'lw': 0.1, 'color': 'k'}, x_transform=lambda x:
                   x*1e3, base_scale_multiplier=1.1)
    ax.set_xlabel('Time (msec)')
    ax.grid(False)

    # Draw lines showing the repeat boundaries                                                                                                            
    for i in range(settings['elicitor_n'] + 2):
        ax.axvline(i * settings['period'] * 1e3, zorder=-1, alpha=0.5)
    return fig


def plot_elicitor_psd(elicitor_db):
    cols = 3
    rows = int(np.ceil(len(elicitor_db) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2), sharex=True,
                             sharey=True)

    for ax, (level, df) in zip(axes.flat, elicitor_db.iterrows()):
        ax.plot(df.iloc[1:], 'k-', lw=0.1)
        ax.grid()
        ax.set_title(f'{level} dB SPL', fontsize=10)

    axes[0, 0].set_xscale('octave', octaves=2)
    axes[0, 0].axis(xmin=1e3, xmax=64e3, ymin=-20, ymax=100)
    for ax in axes[:, 0]:
        ax.set_ylabel('Level (dB SPL)')
    for ax in axes[-1]:
        ax.set_xlabel('Frequency (kHz)')
    fig.tight_layout()
    return fig


def plot_probe_level(probe, silence, probe_psd, silence_psd):
    level = pd.DataFrame({
        'probe': probe_psd.apply(util.rms_rfft_db, axis=1),
        'silence': silence_psd.apply(util.rms_rfft_db, axis=1),
    })

    gs = GridSpec(2, 3)

    fig = plt.Figure(figsize=(12, 8))
    ax_probe = fig.add_subplot(gs[0, :2])
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_probe_psd = fig.add_subplot(gs[1:, :2])

    ax_probe.plot(probe.columns.values * 1e3, probe.values.T, alpha=0.1, color='k', lw=0.25)
    ax_probe.plot(silence.columns.values * 1e3, silence.values.T, alpha=0.1, color='r', lw=0.25)
    ax_probe.set_xlabel('Time (msec)')
    ax_probe.set_ylabel('Signal (V)')

    ax_probe_psd.plot(probe_psd.columns.values, probe_psd.values.T, alpha=0.1, color='k', lw=0.25)
    ax_probe_psd.plot(silence_psd.columns.values, silence_psd.values.T, alpha=0.1, color='r', lw=0.25)

    ax_probe_psd.set_ylabel('Level (dB SPL)')
    ax_probe_psd.set_xlabel('Frequency (kHz)')
    ax_probe_psd.set_xscale('octave')
    ax_probe_psd.axis(xmin=500, xmax=50000, ymin=0)

    p_handle = Line2D([0], [0], color='k')
    s_handle = Line2D([0], [0], color='r')
    ax_probe.legend([p_handle, s_handle], ['Probe', 'Silence'])

    for c, (e, e_df) in iter_colors(level.groupby('elicitor_level')):
        ax_scatter.plot(e_df['probe'], e_df['silence'], 'o', color=c, mec='w', mew=1, label=f'{e}')

    ax_scatter.set_xlabel('Probe (dB SPL)')
    ax_scatter.set_ylabel('Silence (dB SPL)')
    ax_scatter.set_aspect(1, adjustable='datalim')
    ax_scatter.legend(title='Elicitor (dB SPL)', loc='upper left', bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig


def plot_memr(memr_db, memr_level, settings):
    n_repeat = len(memr_db.index.unique('repeat'))
    figure, axes = plt.subplots(2, n_repeat, figsize=(3*n_repeat, 6), sharex='row', sharey='row')

    for i, (repeat, memr_r) in enumerate(memr_db.groupby('repeat')):
        ax = axes[0, i]
        for c, ((_, elicitor), row) in iter_colors(list(memr_r.iterrows())):
            ax.plot(row, color=c, label=f'{elicitor:.0f} dB SPL')
        ax.grid()
        ax.set_xlabel('Frequency (kHz)')
        ax.set_title(f'Repeat {i+1}')
        ax = axes[1, i]
        for label in memr_level.loc[i+1]:
            ax.plot(memr_level.loc[i+1, label], label=label)
        ax.grid()
        ax.set_xlabel('Elicitor level (dB SPL)')

    axes[0, 0].set_xscale('octave')
    axes[0, 0].axis(xmin=settings['probe_fl'], xmax=settings['probe_fh'], ymin=-4, ymax=4)
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
    axes[0, 0].set_ylabel('MEMR (dB)')
    axes[1, -1].legend(loc='lower left', bbox_to_anchor=(1.1, 0))
    axes[1, 0].set_ylabel('MEMR amplitude (dB)')
    figure.tight_layout()
    return figure


def process_interleaved_file(filename, cb, reprocess=False, acoustic_delay=0.75e-3, **kwargs):
    manager = DatasetManager(filename, **kwargs)
    if not reprocess and manager.is_processed(expected_suffixes):
        return
    manager.clear(expected_suffixes)

    with manager.create_cb(cb) as cb:
        fh = InterleavedMEMRFile(filename)
        # Load variables we need from the file
        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()

        fs = fh.probe_microphone.fs
        settings = {
            'period': fh.get_setting('repeat_period'),
            'probe_delay': fh.get_setting('probe_delay'),
            'probe_duration': fh.get_setting('probe_duration'),
            'elicitor_delay': fh.get_setting('elicitor_envelope_start_time'),
            'elicitor_fl': fh.get_setting('elicitor_fl'),
            'elicitor_fh': fh.get_setting('elicitor_fh'),
            'probe_fl': fh.get_setting('probe_fl'),
            'probe_fh': fh.get_setting('probe_fh'),
            'elicitor_n': fh.get_setting('elicitor_n'),
        }

        # First, plot the entire stimulus train. We only plot the positive polarity
        # because if we average in the negative polarity, the noise will cancel
        # out. If we invert then average in the negative polarity, the chirp will
        # cancel out! We just can't win.
        epochs = fh.get_epochs(cb=lambda x: x * 0.5)
        epochs_mean = epochs.xs(1, level='elicitor_polarity').groupby('elicitor_level').mean()
        cb(0.6)

        cb(0.7)

        # Now, load the repeats. This essentially segments the epochs DataFrame
        # into the individual elicitor and probe repeat segments.
        elicitor = fh.get_elicitor()
        elicitor_psd = util.psd_df(elicitor, fs=fs)
        elicitor_spl = elicitor_cal.get_db(elicitor_psd)

        probe = fh.get_probe()

        # Be sure to throw out the last "repeat" (which has a silent period after
        # it rather than another elicitor).
        elicitor_n = settings['elicitor_n']
        elicitor_psd_mean = elicitor_psd.query('repeat < @elicitor_n').groupby('elicitor_level').mean()
        elicitor_spl_mean = elicitor_cal.get_db(elicitor_psd_mean)

        # Now, extract the probe window and the silence following the probe
        # window. The silence will (potentially) be used to estimate artifacts. 

        silence_lb = probe_ub
        silence_ub = probe_ub + settings['probe_duration']
        m = (repeats.columns >= silence_lb) & (repeats.columns < silence_ub)
        silence = repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)

        # Calculate the overall level.
        probe_spl = probe_cal.get_db(util.psd_df(probe, fs=fh.probe_microphone.fs, detrend='constant'))
        silence_spl = probe_cal.get_db(util.psd_df(silence, fs=fh.probe_microphone.fs, detrend='constant'))

        probe_mean = probe.groupby(['repeat', 'elicitor_level']).mean()
        probe_psd_mean = util.psd_df(probe_mean, fh.probe_microphone.fs)
        memr_db = util.db(probe_psd_mean.loc[1:]/probe_psd_mean.loc[0])
        memr_amplitude = pd.DataFrame({
            'negative': -memr_db.loc[:, 5.6e3:11.3e3].min(axis=1),
            'positive': memr_db.loc[:, 8e3:16e3].max(axis=1),
        })

        stim_train_figure = plot_stim_train(epochs_mean, settings)
        elicitor_psd_figure = plot_elicitor_psd(elicitor_spl_mean)
        probe_level_figure = plot_probe_level(probe, silence, probe_spl, silence_spl)
        memr_figure = plot_memr(memr_db, memr_amplitude, settings)

        manager.save_fig(stim_train_figure, 'stimulus train.pdf')
        manager.save_fig(elicitor_psd_figure, 'elicitor PSD.pdf')
        manager.save_fig(probe_level_figure, 'probe level.pdf')
        manager.save_fig(memr_figure, 'MEMR.pdf')
        manager.save_df(memr_db, 'memr.csv')

        plt.close('all')


def process_simultaneous_file(filename, cb, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed(['elicitor PSD.pdf']):
        return
    fh = SimultaneousMEMRFile(filename)

    cal = fh.probe_microphone.get_calibration()
    fs = fh.probe_microphone.fs
    repeats = fh.get_repeats()
    probe_window = fh.get_setting('probe_duration') + 1.5e-3
    probes = repeats.loc[:, :probe_window]
    probe_mean = probes.groupby(['elicitor_level', 'group']).mean()
    probe_spl = cal.get_db(util.psd_df(probe_mean, fs=fs))
    probe_spl_mean = probe_spl.groupby(['elicitor_level', 'group']).mean()
    baseline = probe_spl_mean.xs('baseline', level='group')
    elicitor = probe_spl_mean.xs('elicitor', level='group')
    memr = elicitor - baseline

    epochs = fh.get_epochs()
    onset = fh.get_setting('elicitor_onset')
    duration = fh.get_setting('elicitor_duration')
    elicitor = epochs.loc[:, onset:onset+duration]
    elicitor_waveform = elicitor.loc[1].groupby(['elicitor_level']).mean()
    elicitor_spl = cal.get_db(util.psd_df(elicitor, fs=fs)).dropna(axis='columns')
    elicitor_spl_mean = elicitor_spl.groupby('elicitor_level').mean()

    figure, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    t = probe_mean.columns * 1e3
    for (group, g_df), ax in zip(probe_mean.groupby('group'), axes.flat):
        ax.set_title(f'{group}')
        for level, row in g_df.iterrows():
            ax.plot(t, row, lw=1, label=f'{level[0]} dB SPL')
    for ax in axes[1]:
        ax.set_xlabel('Time (ms)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Signal (V)')
    axes[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper left')
    figure.savefig(manager.get_proc_filename('probe_waveform.pdf'), bbox_inches='tight')

    figure, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for (group, g_df), ax in zip(probe_spl_mean.iloc[:, 1:].groupby('group'), axes.flat):
        ax.set_title(f'{group}')
        for level, row in g_df.iterrows():
            ax.plot(row.index, row, lw=1, label=f'{level[0]} dB SPL')
    for ax in axes[1]:
        ax.set_xlabel('Frequency (kHz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('PSD (dB SPL)')
    axes[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper left')
    axes[0, 0].set_xscale('octave')
    axes[0, 0].axis(xmin=4e3, xmax=32e3)
    figure.savefig(manager.get_proc_filename('probe PSD.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = get_colors(len(memr))
    for c, level, row in zip(colors, memr.iloc[:, 1:].iterrows()):
        ax.plot(row, label=f'{level} dB SPL', color=c)
    ax.set_xscale('octave')
    ax.axis(xmin=4e3, xmax=32e3, ymin=-5, ymax=5)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('MEMR (dB re baseline)')
    figure.savefig(manager.get_proc_filename('MEMR.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_waveform)))
    waterfall_plot(ax, elicitor_waveform, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'})
    figure.savefig(manager.get_proc_filename('elicitor waveform.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_spl_mean)))
    waterfall_plot(ax, elicitor_spl_mean, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'}, scale_method='mean')
    figure.savefig(manager.get_proc_filename('elicitor PSD.pdf'), bbox_inches='tight')
    plt.close('all')


def main_simultaneous_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize simultaneous MEMR data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*memr_simultaneous*',
                  process_simultaneous_file, reprocess=args.reprocess,
                  halt_on_error=args.halt_on_error)


def main_interleaved_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize interleaved MEMR data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*memr_interleaved*',
                  process_interleaved_file, reprocess=args.reprocess)


if __name__ == '__main__':
    main()
