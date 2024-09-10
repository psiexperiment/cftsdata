from pathlib import Path

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import csaps
import pandas as pd
from tqdm import tqdm
from palettable.colorbrewer import qualitative

from psiaudio.plot import iter_colors, waterfall_plot
from psiaudio import util, weighting

from .memr import InterleavedMEMRFile, SimultaneousMEMRFile, SweepMEMRFile
from .util import add_default_options, DatasetManager, process_files


int_expected_suffixes = [
    'elicitor level.csv',
    'MEMR.csv',
    'MEMR.pdf',
    'MEMR_total.csv',
    'MEMR_total.pdf',
    'MEMR_amplitude.csv',
    'MEMR_amplitude_total.csv',
    'probe.pdf',
    'elicitor.pdf',
    'epoch waveform.pdf',
    'processing settings.json',
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

    psd = probe_psd.unstack('repeat')
    probe_psd_valid = psd.loc[valid].stack('repeat')
    probe_psd_invalid = psd.loc[~valid].stack('repeat')

    ax_probe_psd_valid.plot(probe_psd.columns.values, probe_psd_valid.values.T, alpha=alpha, color='seagreen', lw=0.25)
    ax_probe_psd_valid.plot(probe_psd.columns.values, probe_psd_invalid.values.T, alpha=alpha, color='sienna', lw=0.25)
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

    ax_speed.hist(speed.values.flat, bins=100, range=(0, 2), color='k', label=f'Speed')
    ax_speed.axvline(speed_cutoff, ls=':', color='k')
    ax_speed.set_xlabel('Turntable speed (cm/s)')
    ax_speed.set_ylabel('Probe #')
    ax_speed.legend()

    if settings is not None:
        fig.suptitle(f'Probe starship {settings["probe_starship"]}')

    fig.tight_layout()
    return fig


def plot_memr(memr_db, memr_level, settings, mode='conventional'):
    n_repeat = len(memr_db.index.unique('repeat'))
    figsize = (4.5*n_repeat, 7/2*2)
    figure, axes = plt.subplots(2, n_repeat, figsize=figsize, sharex='row',
                                sharey='row', squeeze=False)

    colors = getattr(qualitative, f'Accent_{len(memr_level.columns)}')
    colormap = dict(zip(memr_level.columns.values, colors.mpl_colors))

    for i, (repeat, memr_r) in enumerate(memr_db.groupby('repeat')):
        ax = axes[0, i]
        for c, ((_, elicitor), row) in iter_colors(list(memr_r.iterrows())):
            ax.plot(row, color=c, label=f'{elicitor:.0f} dB SPL')
            for j, (n, (d, lb, ub)) in enumerate(memr_level.attrs['span'].items()):
                if d == 'N':
                    ax.axvspan(lb, ub, ymax=0.05, color=colormap[n], alpha=0.25)
                elif d == 'P':
                    if mode == 'total':
                        ymin, ymax = (0.95, 1.0) if j % 2 else (0.9, 0.95)
                    else:
                        ymin, ymax = 0.95, 1.0
                    ax.axvspan(lb, ub, ymin=ymin, ymax=ymax, color=colormap[n], alpha=0.25)
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

    ps = settings['probe_starship']
    es = settings['elicitor_starship']
    side = 'Ipsilateral' if ps == es else 'Contralateral'
    figure.suptitle(f'{side} MEMR (probe {ps}, elicitor {es})')

    axes[0, 0].set_xscale('octave')
    ymin = -4 if mode == 'conventional' else 0
    axes[0, 0].axis(xmin=settings['probe_fl'], xmax=settings['probe_fh'], ymin=ymin, ymax=4)
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
    axes[0, 0].set_ylabel('MEMR (dB)')
    axes[1, -1].legend(loc='lower left', bbox_to_anchor=(1.1, 0))
    axes[1, 0].set_ylabel('MEMR amplitude (dB)')
    figure.tight_layout()
    return figure


def get_int_settings(fh):
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


def get_sim_settings(fh):
    return {
        'elicitor_onset': float(fh.get_setting('elicitor_onset')),
        'elicitor_duration': float(fh.get_setting('elicitor_duration')),
        'elicitor_fl': fh.get_setting('elicitor_fl'),
        'elicitor_fh': fh.get_setting('elicitor_fh'),
        'probe_fl': fh.get_setting('probe_fl'),
        'probe_fh': fh.get_setting('probe_fh'),
        'weighting': fh.get_setting('elicitor_bandlimited_noise_audiogram_weighting'),
        'elicitor_starship': fh.get_setting('elicitor'),
        'probe_starship': fh.get_setting('probe'),
        'probe_duration': fh.get_setting('probe_duration'),
        'probe_delay': fh.get_setting('probe_delay'),
    }


def calc_memr_amplitude(memr_db, span='conventional'):
    span_options = {
        'conventional': {
            'P1': ('P', 4e3, 8e3),
            'N1': ('N', 5.6e3, 11e3),
            'P2': ('P', 8e3, 16e3),
        },
        'total': {
            'P1': ('P', 4e3, 8e3),
            'P2': ('P', 5.6e3, 11e3),
            'P3': ('P', 8e3, 16e3),
        },
    }
    memr_amplitude = {}
    for (name, (p, lb, ub)) in span_options[span].items():
        if p == 'P':
            memr_amplitude[name] = memr_db.loc[:, lb:ub].max(axis=1)
        elif p == 'N':
            memr_amplitude[name] = -memr_db.loc[:, lb:ub].min(axis=1)
    memr_amplitude = pd.DataFrame(memr_amplitude)
    memr_amplitude.attrs['span'] = span_options[span]
    memr_amplitude.columns.name = 'span'
    return memr_amplitude


def process_interleaved_file(filename, manager, turntable_speed=1.25,
                             min_corr=0.9, **kwargs):
    '''
    Parameters
    ----------
    turntable_speed : {None, float}
        If None, use value saved in settings. Default speed of 1.25 is the
        maximum speed we have been using in our experiments and seems to be
        sufficiently robust to exclude most artifacts.
    min_corr : {None, float}
        If None, use value saved in settings. Rejects the trial if the minimum
        correlation between individual probes in the train is less than this
        value. Lack of correlations usually suggests that an artifact crept
        into the data.
    '''
    with manager.create_cb() as cb:
        fh = InterleavedMEMRFile(filename)
        # Load variables we need from the file
        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()
        settings = get_int_settings(fh)
        fs = fh.probe_microphone.fs

        # First, plot the entire stimulus train. We only plot the positive polarity
        # because if we average in the negative polarity, the noise will cancel
        # out. If we invert then average in the negative polarity, the chirp will
        # cancel out! We just can't win.
        epochs = fh.get_epochs(cb=lambda x: x * 0.5).dropna()
        epochs_mean = epochs.groupby(['elicitor_polarity', 'elicitor_level']).mean()
        cb(0.6)

        speed = fh.get_max_epoch_speed()

        # Now, load the repeats. This essentially segments the epochs DataFrame
        # into the individual elicitor and probe repeat segments.
        elicitor = fh.get_elicitor().dropna()
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

        # This is used for generating some diagnostic plots where we want to
        # keep all the probes, not just the ones that were accepted.
        probe_spl = probe_cal.get_db(util.psd_df(probe, fs=fh.probe_microphone.fs, detrend='constant'))
        silence_spl = probe_cal.get_db(util.psd_df(silence, fs=fh.probe_microphone.fs, detrend='constant'))

        # Pull out only the trials that meet the artifact reject criterion.
        probe_valid = fh.valid_epochs(probe,
                                      turntable_speed=turntable_speed,
                                      min_corr=min_corr)
        silence_valid = probe_valid.reindex(index=probe_valid.index)

        probe_csd = util.csd_df(probe_valid, fs=fh.probe_fs)
        probe_norm = probe_csd.loc[1:] / probe_csd.loc[0]
        flb = settings['probe_fl']
        fub = settings['probe_fh']

        # Calculate MEMR and then calculate the average MEMR across all four
        # repeats and add back to the memr_db
        memr_db = util.db(np.abs(probe_norm)).loc[:, flb:fub].groupby(['repeat', 'elicitor_level']).mean()
        memr_db_mean = memr_db.groupby(['elicitor_level']).mean()
        memr_db_mean = pd.concat([memr_db_mean], keys=['Average'], names=['repeat'])
        memr_db = pd.concat([memr_db_mean, memr_db])
        memr_amplitude = calc_memr_amplitude(memr_db, 'conventional')

        # Now do the same for total MEMR
        memr_db_total = util.db(np.abs(probe_norm - 1) + 1).loc[:, flb:fub].groupby(['repeat', 'elicitor_level']).mean()
        memr_db_total_mean = memr_db_total.groupby(['elicitor_level']).mean()
        memr_db_total_mean = pd.concat([memr_db_total_mean], keys=['Average'], names=['repeat'])
        memr_db_total = pd.concat([memr_db_total_mean, memr_db_total])
        memr_amplitude_total = calc_memr_amplitude(memr_db_total, 'total')

        # Make some diagnostic plots
        # Plot the positive polarity first
        stim_train_figure, ax = plot_stim_train(epochs_mean.loc[1], settings)
        # Now, plot sum of positive and negative to verify they cancel out
        plot_stim_train(epochs_mean.loc[-1] + epochs_mean.loc[1], None, ax=ax, color='r')
        elicitor_psd_figure = plot_elicitor_spl(elicitor_spl_mean, elicitor_level, settings)
        probe_level_figure = plot_probe_level(probe, silence, probe_spl, silence_spl, speed, speed_cutoff=settings['turntable_speed'])

        # Now plot the MEMR
        memr_figure = plot_memr(memr_db, memr_amplitude, settings, 'conventional')
        memr_total_figure = plot_memr(memr_db_total, memr_amplitude_total, settings, 'total')

        manager.save_fig(stim_train_figure, 'epoch waveform.pdf')
        manager.save_fig(elicitor_psd_figure, 'elicitor.pdf')
        manager.save_fig(probe_level_figure, 'probe.pdf')
        manager.save_fig(memr_figure, 'MEMR.pdf')
        manager.save_fig(memr_total_figure, 'MEMR_total.pdf')
        manager.save_df(memr_db.stack().rename('amplitude'), 'MEMR.csv')
        manager.save_df(memr_db_total.stack().rename('amplitude'), 'MEMR_total.csv')
        manager.save_df(memr_amplitude.stack().rename('amplitude'), 'MEMR_amplitude.csv')
        manager.save_df(memr_amplitude_total.stack().rename('amplitude'), 'MEMR_amplitude_total.csv')
        manager.save_df(elicitor_level, 'elicitor level.csv', index=False)
        manager.save_dict({
            'min_corr': min_corr,
            'turntable_speed': turntable_speed,
        }, 'processing settings.json')


sim_expected_suffixes = [
    'valid count.csv',
    'elicitor level.csv',
    'MEMR.csv',
    'MEMR.pdf',
    'MEMR_total.csv',
    'MEMR_total.pdf',
    'MEMR_amplitude.csv',
    'MEMR_amplitude_total.csv',
    'probe.pdf',
    'elicitor.pdf',
    'epoch waveform.pdf',
]


def process_simultaneous_file(filename, manager, turntable_speed=1.25,
                              min_corr=0.9, max_ht2=160, **kwargs):
    with manager.create_cb() as cb:
        fh = SimultaneousMEMRFile(filename)

        settings = get_sim_settings(fh)

        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()

        probe = fh.get_probe(trim=(0, 1e-3))
        valid = fh.get_valid_epoch_mask(turntable_speed=turntable_speed,
                                        min_corr=min_corr, max_ht2=max_ht2)
        probe = probe.xs(0, level='trial')
        valid = valid.xs(0, level='trial')
        valid_count = valid.groupby(['group', 'elicitor_level']) \
            .agg(['size', 'sum', 'mean']).reset_index()

        flb, fub = settings['probe_fl'], settings['probe_fh']
        probe_csd = util.csd_df(probe, fs=fh.probe_fs).loc[:, flb:fub]
        probe_spl = probe_cal.get_db(np.abs(probe_csd))

        probe_csd_mean = probe_csd.loc[valid] \
            .groupby(['group', 'elicitor_polarity', 'elicitor_level']).mean()
        probe_norm = probe_csd_mean.loc['elicitor'] / probe_csd_mean.loc['baseline']
        probe_norm = probe_norm.groupby('elicitor_level').mean()

        memr_db = util.db(np.abs(probe_norm))
        memr_db = pd.concat([memr_db], keys=['Average'], names=['repeat'])
        memr_db_total = util.db(np.abs(probe_norm - 1) + 1)
        memr_db_total = pd.concat([memr_db_total], keys=['Average'], names=['repeat'])

        # Calculate the MEMR amplitude
        memr_amplitude = calc_memr_amplitude(memr_db, 'conventional')
        memr_amplitude_total = calc_memr_amplitude(memr_db_total, 'total')

        elicitor_epochs = fh.get_epochs(signal_name='elicitor_microphone') \
            .xs(0, level='trial').reset_index('t0', drop=True)
        probe_epochs = fh.get_epochs(signal_name='probe_microphone') \
            .xs(0, level='trial').reset_index('t0', drop=True)

        # Plot probe waveform and PSD for review purposes
        probe_figure, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        for row, group in zip(axes, ('baseline', 'elicitor')):
            x = probe[valid].xs(group, level='group')
            row[0].plot(x.columns.values * 1e3, x.values.T, alpha=0.25, color='k', lw=0.25);
            try:
                x = probe[~valid].xs(group, level='group')
                row[0].plot(x.columns.values * 1e3, x.values.T, alpha=0.25, color='r', lw=0.25);
            except KeyError:
                pass

            x = probe_spl[valid].xs(group, level='group')
            row[1].plot(x.columns.values, x.values.T, alpha=0.25, color='k', lw=0.25);
            try:
                x = probe_spl[~valid].xs(group, level='group')
                row[1].plot(x.columns.values, x.values.T, alpha=0.25, color='r', lw=0.25);
            except KeyError:
                pass

            row[1].set_xscale('octave')
            row[1].axis(xmin=500, xmax=50000, ymin=0)
            row[0].set_xlabel('Time (ms)')
            row[0].set_ylabel('Amplitude')
            row[1].set_xlabel('Frequency (kHz)')
            row[1].set_ylabel('Level (dB SPL)')
            row[0].set_title(f'{group} probe waveform')
            row[1].set_title(f'{group} probe PSD')

        # Plot the epoch trains to check for any potential issues
        e_mean = elicitor_epochs.loc[-1] + elicitor_epochs.loc[1]
        e_pos = elicitor_epochs.loc[1]
        epoch_figure, axes = plt.subplots(1, 2, figsize=(12, 1 * len(e_mean)))
        waterfall_plot(axes[0], e_pos, 'elicitor_level', plotkw={'lw': 0.1, 'color': 'k'})
        waterfall_plot(axes[0], e_mean, 'elicitor_level', plotkw={'lw': 0.1, 'color': 'r'})
        waterfall_plot(axes[1], probe_epochs, 'elicitor_level', plotkw={'lw': 0.1, 'color': 'k'})
        axes[0].set_xlabel('Time (s)')
        axes[1].set_xlabel('Time (s)')
        axes[0].set_title(f'Elicitor {settings["elicitor_starship"]}')
        axes[1].set_title(f'Probe {settings["probe_starship"]}')

        # Code to generate the elicitor level plot
        o, d = settings['elicitor_onset'], settings['elicitor_duration']
        lb, ub = settings['elicitor_fl'], settings['elicitor_fh']

        elicitor_waveform = elicitor_epochs.loc[:, o:o+d]
        elicitor_psd = util.psd_df(elicitor_waveform, fs=fh.elicitor_fs).groupby('elicitor_level').mean()
        elicitor_spl = elicitor_cal.get_db(elicitor_psd)
        subset = elicitor_spl.loc[:, lb:ub]

        w = weighting.load(subset.columns, settings['weighting'])
        def calc_level(x):
            #nonlocal w
            return pd.Series({
                'actual': util.rms_rfft_db(x),
                'weighted': util.rms_rfft_db(x - w),
            })
        elicitor_level = subset.apply(calc_level, axis=1)
        elicitor_level['requested'] = elicitor_level.index.get_level_values('elicitor_level')
        elicitor_level = elicitor_level.reset_index(drop=True)
        elicitor_figure = plot_elicitor_spl(elicitor_spl, elicitor_level, settings)

        memr_figure = plot_memr(memr_db, memr_amplitude, settings, 'conventional')
        memr_total_figure = plot_memr(memr_db_total, memr_amplitude_total, settings, 'total')

        manager.save_df(valid_count, 'valid count.csv', index=False)
        manager.save_df(elicitor_level, 'elicitor level.csv', index=False)
        manager.save_df(memr_db.stack().rename('amplitude'), 'MEMR.csv')
        manager.save_df(memr_db_total.stack().rename('amplitude'), 'MEMR_total.csv')
        manager.save_fig(memr_figure, 'MEMR.pdf')
        manager.save_fig(memr_total_figure, 'MEMR_total.pdf')
        manager.save_df(memr_amplitude.stack().rename('amplitude'), 'MEMR_amplitude.csv')
        manager.save_df(memr_amplitude_total.stack().rename('amplitude'), 'MEMR_amplitude_total.csv')
        manager.save_fig(probe_figure, 'probe.pdf')
        manager.save_fig(elicitor_figure, 'elicitor.pdf')
        manager.save_fig(epoch_figure, 'epoch waveform.pdf')


###############################################################################
# MEMR sweep
###############################################################################
def plot_sweep_probe(probe, probe_spl):
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    probe_mean = probe.mean(axis=0)
    probe_spl_mean = probe_spl.groupby('repeat').mean()

    axes[0].plot(probe_mean.index.values * 1e3, probe_mean, 'k', lw=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (V)')

    axes[1].plot(probe_spl_mean.iloc[:, 1:].T, 'k', lw=0.1, alpha=0.1);
    axes[1].set_xscale('octave')
    axes[1].axis(xmin=4e3*.75, xmax=32e3/0.75)
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('Level (dB SPL)')

    return figure


def sweep_elicitor_psd(elicitor, fs, window_size=50e-3):
    window_size = int(window_size * fs)
    window_step = window_size // 10

    elicitor_psd = []
    for epoch in elicitor.values:
        sv = sliding_window_view(epoch, window_size)
        p = util.psd_df(sv[::window_step], fs=fs)
        p.index = pd.Index(np.arange(len(p)) * window_step / fs, name='time')
        elicitor_psd.append(p)

    elicitor_psd = pd.concat(elicitor_psd, keys=np.arange(len(elicitor_psd)), names=['trial'])
    return elicitor_psd.groupby('time').mean()


def plot_sweep_elicitor(elicitor_mean, elicitor_spl, settings):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    lb, ub = settings['elicitor_fl'], settings['elicitor_fh']

    ax = axes[2]
    i = len(elicitor_spl) // 2
    ax.plot(elicitor_spl.iloc[i, 1:], 'k-', lw=0.1)
    ax.set_xscale('octave')
    ax.axis(xmin=1e3, xmax=64e3, ymin=0)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Level (dB SPL)')
    ax.axvspan(lb, ub, color='lightblue')

    level = elicitor_spl.loc[:, lb*.9:ub/.9].apply(util.rms_rfft_db, axis=1)
    ramp_rate = settings['ramp_rate']

    ax = axes[1]
    peak_time = level.idxmax()
    peak = level.loc[peak_time]
    ax.axvline(peak_time, ls=':', color='seagreen')
    ax.axhline(peak, ls=':', color='seagreen')
    y = np.abs(level.index - peak_time) * -ramp_rate + peak
    ax.plot(level.index, y, color='orange', label='Expected', lw=0.5)
    ax.plot(level, color='k', label='Actual', lw=0.5)
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Level (dB SPL)')
    ax.legend()

    ax = axes[0]
    ax.plot(elicitor_mean.loc[1], 'k', lw=0.1, label='Average of + polarity')
    ax.plot(0.5 * elicitor_mean.loc[1] + 0.5 * elicitor_mean.loc[-1], 'r', lw=0.1, label='Average of +/- polarity')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Measured amplitude (V)')

    ps = settings['probe_starship']
    es = settings['elicitor_starship']
    side = 'Ipsilateral' if ps == es else 'Contralateral'
    figure.suptitle(f'{side} MEMR (probe {ps}, elicitor {es})')

    return figure


def sweep_artifact_reject(x, q=25, multiplier=1.5):
    lb, ub = np.percentile(x, [q, 100-q])
    iqr = ub - lb
    return (x < (lb - multiplier * iqr)) | (x > (ub + multiplier * iqr))


def sweep_get_weights(x):
    w = np.ones_like(x)
    r = sweep_artifact_reject(x)
    w[r] = 0.1
    return w


def sweep_detrend(s, fs, smooth=0.9999999):
    x = np.arange(len(s)) / fs
    y = np.vstack([np.real(s), np.imag(s)])
    w = sweep_get_weights(y[0])
    spline = csaps.CubicSmoothingSpline(x, y, w, smooth=smooth)
    ys = y - spline(x) + y.mean(axis=1, keepdims=True)
    return pd.Series(ys[0] + 1j * ys[1], index=s.index)


def sweep_csaps_smooth(a, smooth=0.1):
    x = a.index.values
    yr = np.real(a.values)
    yi = np.imag(a.values)
    y = np.vstack([yr, yi])
    spline = csaps.CubicSmoothingSpline(x, y, smooth=smooth)
    ys = spline(x)
    return ys[0] + 1j * ys[1]


def plot_sweep_memr(memr, memr_total):
    figure, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    axes[0, 0].plot(memr.loc[:, :16000], alpha=0.1, lw=0.5, color='k');
    axes[0, 0].set_ylabel('Change (dB)')
    axes[0, 0].set_xlabel('Click number')

    axes[0, 1].plot(memr_total.loc[:, :16000], alpha=0.1, lw=0.5, color='k');
    axes[0, 1].set_ylabel('Total change (dB)')
    axes[0, 1].set_xlabel('Click number')

    i = len(memr) // 2
    axes[1, 0].plot(memr.T, lw=0.5, alpha=0.1, color='k');
    axes[1, 0].plot(memr.loc[0], lw=0.5, alpha=1, color='orange');
    axes[1, 0].plot(memr.loc[i], lw=0.5, alpha=1, color='blue');
    axes[1, 0].set_xscale('octave')
    axes[1, 0].set_ylabel('Change (dB)')
    axes[1, 0].set_xlabel('Frequency (kHz)')

    axes[1, 1].plot(memr_total.T, lw=0.5, alpha=0.1, color='k');
    axes[1, 1].plot(memr_total.loc[0], lw=0.5, alpha=1, color='orange', label='Baseline');
    axes[1, 1].plot(memr_total.loc[i], lw=0.5, alpha=1, color='blue', label='Peak elicitor');
    axes[1, 1].set_xscale('octave')
    axes[1, 1].legend()
    axes[1, 1].set_ylabel('Total change (dB)')
    axes[1, 1].set_xlabel('Frequency (kHz)')

    return figure


def plot_sweep_diagnostics(f_max, r_csd, r_csd_dt, r_csd_sm):
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    s = r_csd.loc[:, f_max].values
    axes[0].plot(util.db(np.abs(s)), label='raw')
    s = r_csd_dt.loc[:, f_max].values
    axes[0].plot(util.db(np.abs(s)), label='detrended')
    axes[0].legend()
    axes[0].set_xlabel('Trial # x probe #')
    axes[0].set_ylabel('Amplitude (dB)')

    x = util.db(np.abs(r_csd.loc[:, f_max].groupby('repeat').mean()))
    axes[1].plot(x, label='raw')
    x = util.db(np.abs(r_csd_dt.loc[:, f_max].groupby('repeat').mean()))
    plt.plot(x, label='detrended')
    x = util.db(np.abs(r_csd_sm.loc[:, f_max].groupby('repeat').mean()))
    axes[1].plot(x, label='smoothed')
    axes[1].set_xlabel('Probe # (average across trials)')
    axes[1].set_ylabel('Amplitude (dB)')
    axes[1].legend()
    return figure


sweep_expected_suffixes = [
    'probe.pdf',
    'elicitor.pdf',
    'MEMR.csv',
    'MEMR_total.csv',
    'MEMR.pdf',
    'MEMR_block.pdf',
    'diagnostics.pdf',
]


def get_sweep_settings(fh):
    return {
        'trial_duration': fh.get_setting('trial_duration'),
        'probe_starship': fh.get_setting('probe'),
        'elicitor_starship': fh.get_setting('elicitor'),
        'ramp_rate': fh.get_setting('ramp_rate'),
        'elicitor_fl': fh.get_setting('elicitor_fl'),
        'elicitor_fh': fh.get_setting('elicitor_fh'),
        'elicitor_min_level': fh.get_setting('min_level', 0),
        'elicitor_max_level': fh.get_setting('max_level'),
        'probe_rate': fh.get_setting('probe_rate'),
        'probe_n': fh.get_setting('probe_n'),
    }


def csd_to_swept_memr(r_csd_dt):
    r_csd_mean = r_csd_dt.groupby('repeat').mean()
    r_csd_sm = r_csd_mean.apply(sweep_csaps_smooth)
    baseline = r_csd_sm.iloc[[0, 1, 2, -3, -2, -1]].mean()
    r_csd_norm = r_csd_sm / baseline
    memr_db = util.db(np.abs(r_csd_norm))
    memr_db_total = util.db(np.abs(r_csd_norm - 1) + 1)
    return r_csd_sm, memr_db, memr_db_total


def process_sweep_file(filename, manager, turntable_speed=1.25,
                       trials_per_block=5, **kwargs):
    '''
    Parameters
    ----------
    turntable_speed : {None, float}
        If None, use value saved in settings. Default speed of 1.25 is the
        maximum speed we have been using in our experiments and seems to be
        sufficiently robust to exclude most artifacts.
    '''
    with manager.create_cb() as cb:
        fh = SweepMEMRFile(filename)
        probe_epoch = fh.get_epochs()
        elicitor_epoch = fh.get_epochs(signal_name='elicitor_microphone')
        probe = fh.get_probe()
        probe_cal = fh.probe_microphone.get_calibration()
        elicitor_cal = fh.elicitor_microphone.get_calibration()

        probe_spl = probe_cal.get_db(util.psd_df(probe, fs=fh.elicitor_fs))
        probe_fig = plot_sweep_probe(probe, probe_spl)

        settings = get_sweep_settings(fh)

        elicitor_mean = elicitor_epoch.groupby('elicitor_polarity').mean()
        elicitor_psd = sweep_elicitor_psd(elicitor_epoch, fh.elicitor_fs)
        elicitor_spl = elicitor_cal.get_db(elicitor_psd)
        elicitor_fig = plot_sweep_elicitor(elicitor_mean, elicitor_spl, settings)

        r_csd = util.csd_df(probe, fs=fh.probe_fs, window='hann').loc[:, 4e3:32e3]
        r_csd_dt = r_csd.apply(sweep_detrend, fs=fh.probe_fs)

        block_figs = []
        n_blocks = int(np.ceil(fh.get_setting('trial_n') / trials_per_block))
        for i in range(n_blocks):
            lb = i * trials_per_block
            ub = lb + trials_per_block
            block = r_csd_dt.loc[lb:ub]
            _, block_memr_db, block_memr_db_total = csd_to_swept_memr(block)
            fig = plot_sweep_memr(block_memr_db, block_memr_db_total)
            fig.suptitle(f'MEMR subset (trials {lb+1} to {ub+1})')
            block_figs.append(fig)

        r_csd_sm, memr_db, memr_db_total = csd_to_swept_memr(r_csd_dt)
        memr_fig = plot_sweep_memr(memr_db, memr_db_total)
        f_max = memr_db_total.loc[len(memr_db_total) // 2, :16e3].idxmax()
        dx_fig = plot_sweep_diagnostics(f_max, r_csd, r_csd_dt, r_csd_sm)

        probe_times = np.arange(settings['probe_n'])/settings['probe_rate']
        probe_times -= settings['trial_duration']/2
        lb, ub = settings['elicitor_min_level'], settings['elicitor_max_level']
        elicitor_level = (ub-lb) - np.abs(probe_times) * settings['ramp_rate'] + settings['elicitor_min_level']
        elicitor_level = pd.Series(elicitor_level, name='elicitor_level')
        elicitor_level.index.name = 'repeat'

        memr_db = memr_db.join(elicitor_level).set_index('elicitor_level', append=True)
        memr_db.columns.name = 'frequency'
        memr_db_total = memr_db_total.join(elicitor_level).set_index('elicitor_level', append=True)
        memr_db_total.columns.name = 'frequency'

        manager.save_df(memr_db.stack().rename('amplitude'), 'MEMR.csv')
        manager.save_df(memr_db_total.stack().rename('amplitude'), 'MEMR_total.csv')
        manager.save_fig(probe_fig, 'probe.pdf')
        manager.save_fig(elicitor_fig, 'elicitor.pdf')
        manager.save_fig(memr_fig, 'MEMR.pdf')
        manager.save_figs(block_figs, 'MEMR_block.pdf')
        manager.save_fig(dx_fig, 'diagnostics.pdf')


def main_valero():
    import argparse
    parser = argparse.ArgumentParser('Summarize Valero MEMR data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_simultaneous*',
                  fn=process_simultaneous_file,
                  expected_suffixes=sim_expected_suffixes, **args)


def main_keefe():
    import argparse
    parser = argparse.ArgumentParser('Summarize Keefe MEMR data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_interleaved*',
                  fn=process_interleaved_file,
                  expected_suffixes=int_expected_suffixes, **args)


def main_sweep():
    import argparse
    parser = argparse.ArgumentParser('Summarize MEMR sweep data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_sweep*',
                  fn=process_sweep_file,
                  expected_suffixes=sweep_expected_suffixes, **args)
