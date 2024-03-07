from pathlib import Path

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from palettable.colorbrewer import qualitative

from psiaudio.plot import iter_colors, waterfall_plot
from psiaudio import util, weighting

from .memr import InterleavedMEMRFile, SimultaneousMEMRFile
from .util import add_default_options, DatasetManager, process_files


int_expected_suffixes = [
    'stimulus train.pdf',
    'elicitor PSD.pdf',
    'probe level.pdf',
    'MEMR.pdf',
    'elicitor level.csv',
    'MEMR.csv',
]


def plot_stim_train(epochs, settings=None, ax=None, color='k'):
    if ax is None:
        figsize = 6, 1 * len(epochs)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    waterfall_plot(ax, epochs, 'elicitor_level', scale_method='max',
                   plotkw={'lw': 0.1, 'color': color}, x_transform=lambda x:
                   x*1e3, base_scale_multiplier=1.1)
    ax.set_xlabel('Time (msec)')
    ax.grid(False)

    # Draw lines showing the repeat boundaries
    if settings is not None:
        for i in range(settings['elicitor_n'] + 2):
            ax.axvline(i * settings['period'] * 1e3, zorder=-1, alpha=0.5)

    if fig is not None:
        return fig, ax


def plot_elicitor_spl(elicitor_db, elicitor_levels, settings):
    n_axes = len(elicitor_db)
    cols = 3
    rows = max(2, int(np.ceil(n_axes / cols)))

    gs = GridSpec(rows, 5)
    fig = plt.Figure(figsize=(cols*2*2, rows*2))
    axes = []
    for i in range(n_axes):
        c = i % cols
        r = i // cols
        if len(axes) != 0:
            ax = fig.add_subplot(gs[r, c], sharex=axes[0], sharey=axes[0])
        else:
            ax = fig.add_subplot(gs[r, c])
        if c == 0:
            ax.set_ylabel('Level (dB SPL)')
        if r == (rows-1):
            ax.set_xlabel('Frequency (kHz)')
        axes.append(ax)

    level_ax = fig.add_subplot(gs[:2, 3:])

    for ax, (level, df) in zip(axes, elicitor_db.iterrows()):
        ax.plot(df.iloc[1:], 'k-', lw=0.1)
        ax.grid()
        ax.set_title(f'{level} dB SPL', fontsize=10)
        if settings is not None:
            ax.axvspan(settings['elicitor_fl'], settings['elicitor_fh'], color='lightblue')

    if settings is not None:
        fig.suptitle(f'Elicitor starship {settings["elicitor_starship"]}')

    lb = elicitor_levels['weighted'].min()
    ub = elicitor_levels['weighted'].max()
    level_ax.plot([lb, ub], [lb, ub], '-', color='seagreen')
    kw = {'mec': 'w', 'mew': 1}
    # Don't plot the weighted values if there was no weighting applied.
    w = settings['weighting']
    # np.isnan does not work on strings, so just convert np.nan to string for the check.
    if str(w) != 'nan':
        level_ax.plot(elicitor_levels['requested'], elicitor_levels['weighted'], 'o', color='k', label=f'Level (dB re {w})', **kw)
    level_ax.plot(elicitor_levels['requested'], elicitor_levels['actual'], 'o', color='0.5', label='Level (dB SPL)', **kw)
    level_ax.grid()
    level_ax.set_xlabel('Requested level')
    level_ax.set_ylabel('Measured level')
    level_ax.legend()

    axes[0].set_xscale('octave', octaves=2)
    axes[0].axis(xmin=1e3, xmax=64e3, ymin=-20, ymax=80)
    fig.tight_layout()
    return fig


def plot_probe_level(probe, silence, probe_psd, silence_psd, speed,
                     speed_cutoff=0.5, alpha=0.25, settings=None):
    gs = GridSpec(3, 3)

    fig = plt.Figure(figsize=(12, 12))
    ax_probe = fig.add_subplot(gs[0, :2])
    ax_probe_psd = fig.add_subplot(gs[1, :2])
    ax_probe_psd_valid = fig.add_subplot(gs[2, :2])

    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_speed = fig.add_subplot(gs[1, 2])

    ax_probe.plot(probe.columns.values * 1e3, probe.values.T, alpha=alpha, color='k', lw=0.25)
    ax_probe.plot(silence.columns.values * 1e3, silence.values.T, alpha=alpha, color='r', lw=0.25)
    ax_probe.set_xlabel('Time (msec)')
    ax_probe.set_ylabel('Signal (V)')

    ax_probe_psd.plot(probe_psd.columns.values, probe_psd.values.T, alpha=alpha, color='k', lw=0.25)
    ax_probe_psd.plot(silence_psd.columns.values, silence_psd.values.T, alpha=alpha, color='r', lw=0.25)
    ax_probe_psd.set_ylabel('Level (dB SPL)')
    ax_probe_psd.set_xlabel('Frequency (kHz)')
    ax_probe_psd.set_xscale('octave')
    ax_probe_psd.axis(xmin=500, xmax=50000, ymin=0)
    p_handle = Line2D([0], [0], color='k')
    s_handle = Line2D([0], [0], color='r')
    ax_probe.legend([p_handle, s_handle], ['Probe', 'Silence'])

    valid = speed < speed_cutoff
    ax_probe_psd_valid.plot(probe_psd.columns.values, probe_psd.loc[valid].values.T, alpha=alpha, color='seagreen', lw=0.25)
    ax_probe_psd_valid.plot(probe_psd.columns.values, probe_psd.loc[~valid].values.T, alpha=alpha, color='sienna', lw=0.25)
    ax_probe_psd_valid.set_ylabel('Level (dB SPL)')
    ax_probe_psd_valid.set_xlabel('Frequency (kHz)')
    ax_probe_psd_valid.set_xscale('octave')
    ax_probe_psd_valid.axis(xmin=500, xmax=50000, ymin=0)
    p_handle = Line2D([0], [0], color='seagreen')
    s_handle = Line2D([0], [0], color='sienna')
    ax_probe_psd_valid.legend([p_handle, s_handle], ['Valid', 'Reject'])

    level = pd.DataFrame({
        'probe': probe_psd.apply(util.rms_rfft_db, axis=1),
        'silence': silence_psd.apply(util.rms_rfft_db, axis=1),
    })
    for c, (e, e_df) in iter_colors(level.groupby('elicitor_level')):
        ax_scatter.plot(e_df['probe'], e_df['silence'], 'o', color=c, mec='w', mew=1, label=f'{e}')
    ax_scatter.set_xlabel('Probe (dB SPL)')
    ax_scatter.set_ylabel('Silence (dB SPL)')
    ax_scatter.set_aspect(1, adjustable='datalim')
    ax_scatter.legend(title='Elicitor (dB SPL)', loc='upper left', bbox_to_anchor=(1, 1))

    speed_baseline = speed.loc[0]
    speed_elicitor = speed.loc[1:]
    pct_baseline = (speed_baseline < speed_cutoff).mean() * 100
    pct_elicitor = (speed_elicitor < speed_cutoff).mean() * 100
    ax_speed.hist(speed.loc[1:].values.flat, bins=100, range=(0, 2), color='0.5', label=f'Elicitor ({pct_elicitor:.0f}% valid)')
    ax_speed.hist(speed.loc[0].values.flat, bins=100, range=(0, 2), color='k', label=f'Baseline ({pct_baseline:.0f}% valid)')
    ax_speed.axvline(speed_cutoff, ls=':', color='k')
    ax_speed.set_xlabel('Turntable speed (cm/s)')
    ax_speed.set_ylabel('Probe #')
    ax_speed.legend()

    if settings is not None:
        fig.suptitle(f'Probe starship {settings["probe_starship"]}')

    fig.tight_layout()
    return fig


def plot_memr(memr_db, memr_level, settings):
    n_repeat = len(memr_db.index.unique('repeat'))
    figure, axes = plt.subplots(3, n_repeat, figsize=(3.5*n_repeat, 7/2*3), sharex='row', sharey='row')

    colors = getattr(qualitative, f'Accent_{len(memr_level.columns)}')
    colormap = dict(zip(memr_level.columns.values, colors.mpl_colors))

    for i, (repeat, memr_r) in enumerate(memr_db.groupby('repeat')):
        ax = axes[0, i]
        for c, ((_, elicitor), row) in iter_colors(list(memr_r.iterrows())):
            ax.plot(row, color=c, label=f'{elicitor:.0f} dB SPL')
            for n, (d, lb, ub) in memr_level.attrs['span'].items():
                if d == 'N':
                    ax.axvspan(lb, ub, ymax=0.05, color=colormap[n], alpha=0.25)
                elif d == 'P':
                    ax.axvspan(lb, ub, ymin=0.95, color=colormap[n], alpha=0.25)
                else:
                    raise ValueError('Unsupported peak type')
        ax.grid()
        ax.set_xlabel('Frequency (kHz)')
        ax.set_title(f'Repeat {repeat}')
        ax = axes[1, i]
        for label in memr_level.loc[repeat]:
            ax.plot(memr_level.loc[repeat, label], label=label, color=colormap[label])
        ax.grid()
        ax.set_xlabel('Elicitor level (dB SPL)')

    for label, ax in zip(memr_level, axes[2]):
        for c, (r, r_df) in iter_colors(memr_level[label].groupby('repeat')):
            r_df = r_df.reset_index()
            ax.plot(r_df['elicitor_level'], r_df[label], label=f'Repeat {r}', color=c)
            ax.set_title(label)
        ax.legend()
        ax.grid()
        ax.set_xlabel('Elicitor level (dB SPL)')
    axes[2, 0].set_ylabel('MEMR amplitude (dB)')

    for ax in axes[2]:
        if len(ax.lines) == 0:
            ax.remove()

    ps = settings['probe_starship']
    es = settings['elicitor_starship']
    side = 'Ipsilateral' if ps == es else 'Contralateral'
    figure.suptitle(f'{side} MEMR (probe {ps}, elicitor {es})')

    axes[0, 0].set_xscale('octave')
    axes[0, 0].axis(xmin=settings['probe_fl'], xmax=settings['probe_fh'], ymin=-4, ymax=4)
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
    axes[0, 0].set_ylabel('MEMR (dB)')
    axes[1, -1].legend(loc='lower left', bbox_to_anchor=(1.1, 0))
    axes[1, 0].set_ylabel('MEMR amplitude (dB)')
    figure.tight_layout()
    return figure


def get_settings(fh):
    return {
        'period': fh.get_setting('repeat_period'),
        'probe_delay': fh.get_setting('probe_delay'),
        'probe_duration': fh.get_setting('probe_duration'),
        'elicitor_delay': fh.get_setting('elicitor_envelope_start_time'),
        'elicitor_fl': fh.get_setting('elicitor_fl'),
        'elicitor_fh': fh.get_setting('elicitor_fh'),
        'probe_fl': fh.get_setting('probe_fl'),
        'probe_fh': fh.get_setting('probe_fh'),
        'elicitor_n': fh.get_setting('elicitor_n'),
        'weighting': fh.get_setting('elicitor_bandlimited_noise_audiogram_weighting'),
        'turntable_speed': fh.get_setting('max_turntable_speed'),
        'probe_starship': fh.get_setting('probe'),
        'elicitor_starship': fh.get_setting('elicitor'),
        'trial_n': fh.get_setting('trial_n'),
    }


def process_interleaved_file(filename, manager, acoustic_delay=0.75e-3,
                             turntable_speed=1, **kwargs):
    '''
    Parameters
    ----------
    turntable_speed : {None, float}
        If None, use value saved in settings.
    '''
    with manager.create_cb() as cb:
        fh = InterleavedMEMRFile(filename)
        # Load variables we need from the file
        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()
        settings = get_settings(fh)
        fs = fh.probe_microphone.fs

        # First, plot the entire stimulus train. We only plot the positive polarity
        # because if we average in the negative polarity, the noise will cancel
        # out. If we invert then average in the negative polarity, the chirp will
        # cancel out! We just can't win.
        epochs = fh.get_epochs(cb=lambda x: x * 0.5)
        epochs_mean = epochs.groupby(['elicitor_polarity', 'elicitor_level']).mean()
        cb(0.6)

        # Load the turntable speed and find maximum across entire repeat.
        speed = fh.get_speed().max(axis=1)

        # If turntable_speed is not set, load from the settings.
        if turntable_speed is None:
            turntable_speed = settings['turntable_speed']
        valid = speed < turntable_speed

        # Now, load the repeats. This essentially segments the epochs DataFrame
        # into the individual elicitor and probe repeat segments.
        elicitor = fh.get_elicitor()
        elicitor_psd = util.psd_df(elicitor, fs=fs)
        elicitor_spl = elicitor_cal.get_db(elicitor_psd)

        # Be sure to throw out the last "repeat" (which has a silent period after
        # it rather than another elicitor).
        elicitor_n = settings['elicitor_n']
        elicitor_psd_mean = elicitor_psd.query(f'repeat < {elicitor_n}').groupby('elicitor_level').mean()
        elicitor_spl_mean = elicitor_cal.get_db(elicitor_psd_mean)

        # Calculate the weighted and unweighted elicitor level
        lb = settings['elicitor_fl']
        ub = settings['elicitor_fh']
        subset = elicitor_spl.query(f'repeat < {elicitor_n}').loc[:, lb:ub]
        w = weighting.load(subset.columns, settings['weighting'])
        def calc_level(x):
            nonlocal w
            return pd.Series({
                'actual': util.rms_rfft_db(x),
                'weighted': util.rms_rfft_db(x - w),
            })
        elicitor_level = subset.apply(calc_level, axis=1)
        elicitor_level['requested'] = elicitor_level.index.get_level_values('elicitor_level')
        elicitor_level = elicitor_level.reset_index(drop=True)

        # Now, extract the probe window and the silence following the probe
        # window. The silence will (potentially) be used to estimate artifacts.
        probe = fh.get_probe()
        silence = fh.get_silence()

        # Calculate the overall level.
        probe_spl = probe_cal.get_db(util.psd_df(probe, fs=fh.probe_microphone.fs, detrend='constant'))
        silence_spl = probe_cal.get_db(util.psd_df(silence, fs=fh.probe_microphone.fs, detrend='constant'))

        trial_n = int(settings['trial_n'] / 2)
        if (trial_n * 2) != settings['trial_n']:
            raise ValueError('Unequal number of positive and negative polarity trials')

        grouping = ['repeat', 'elicitor_level', 'elicitor_polarity']
        probe_valid = probe.loc[valid].groupby(grouping).apply(lambda x: x.iloc[:trial_n])
        probe_valid_mean = probe_valid.groupby(['repeat', 'elicitor_level']).mean()
        probe_valid_psd_mean = util.psd_df(probe_valid_mean, fh.probe_microphone.fs)
        memr_db = util.db(probe_valid_psd_mean.loc[1:] / probe_valid_psd_mean.loc[0])

        # Calculate the average MEMR across all four repeats.
        memr_db_mean = memr_db.groupby(['elicitor_level']).mean()
        memr_db_mean = pd.concat([memr_db_mean], keys=['Average'], names=['repeat'])
        memr_db = pd.concat([memr_db_mean, memr_db])

        # Caluclate the MEMR amplitude
        spans = {
            'P1': ('P', 4e3, 8e3),
            'N1': ('N', 5.6e3, 11e3),
            'P2': ('P', 8e3, 16e3),
            'N2': ('N', 11.3e3, 22.6e3),
            'P3': ('P', 16e3, 32e3),
        }

        memr_amplitude = {}
        for (name, (p, lb, ub)) in spans.items():
            if p == 'P':
                memr_amplitude[name] = memr_db.loc[:, lb:ub].max(axis=1)
            elif p == 'N':
                memr_amplitude[name] = -memr_db.loc[:, lb:ub].min(axis=1)
        memr_amplitude = pd.DataFrame(memr_amplitude)
        memr_amplitude.attrs['span'] = spans

        # Plot the positive polarity first
        stim_train_figure, ax = plot_stim_train(epochs_mean.loc[1], settings)
        # Now, plot sum of positive and negative to verify they cancel out
        plot_stim_train(epochs_mean.loc[-1] + epochs_mean.loc[1], None, ax=ax, color='r')

        elicitor_psd_figure = plot_elicitor_spl(elicitor_spl_mean, elicitor_level, settings)
        probe_level_figure = plot_probe_level(probe, silence, probe_spl, silence_spl, speed, speed_cutoff=settings['turntable_speed'])
        memr_figure = plot_memr(memr_db, memr_amplitude, settings)

        manager.save_fig(stim_train_figure, 'stimulus train.pdf')
        manager.save_fig(elicitor_psd_figure, 'elicitor PSD.pdf')
        manager.save_fig(probe_level_figure, 'probe level.pdf')
        manager.save_fig(memr_figure, 'MEMR.pdf')
        manager.save_df(memr_db, 'MEMR.csv')
        manager.save_df(elicitor_level, 'elicitor level.csv')

        plt.close('all')

sim_expected_suffixes = [
    'elicitor PSD.pdf'
]

def process_simultaneous_file(filename, manager, **kwargs):
    with manager.create_cb() as cb:
        fh = SimultaneousMEMRFile(filename)

        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()

        fs = fh.probe_microphone.fs

        epochs = fh.get_epochs()
        repeats = fh.get_repeats()

        probe_window = fh.get_setting('probe_duration') + 1.5e-3
        probes = repeats.loc[:, :probe_window]
        probe_mean = probes.groupby(['elicitor_level', 'group']).mean()
        probe_spl = probe_cal.get_db(util.psd_df(probe_mean, fs=fs))
        probe_spl_mean = probe_spl.groupby(['elicitor_level', 'group']).mean()
        baseline = probe_spl_mean.xs('baseline', level='group')
        elicitor = probe_spl_mean.xs('elicitor', level='group')
        memr = elicitor - baseline

        onset = float(fh.get_setting('elicitor_onset'))
        duration = float(fh.get_setting('elicitor_duration'))
        elicitor = epochs.loc[:, onset:onset+duration]
        elicitor_waveform = elicitor.loc[1].groupby(['elicitor_level']).mean()
        elicitor_spl = elicitor_cal.get_db(util.psd_df(elicitor, fs=fs)).dropna(axis='columns')
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
        for c, (level, row) in iter_colors(list(memr.iloc[:, 1:].iterrows())):
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
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_simultaneous*',
                  fn=process_simultaneous_file,
                  expected_suffixes=sim_expected_suffixes, **args)


def main_interleaved_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize interleaved MEMR data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_interleaved*',
                  fn=process_interleaved_file,
                  expected_suffixes=int_expected_suffixes, **args)
