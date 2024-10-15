import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psiaudio import util

from .dpoae import DPOAEFile, isodp_th_criterions
from .util import add_default_options, DatasetManager, process_files


dpoae_expected_suffixes = [
    'io.csv',
    'th.csv',
    'io.pdf',
    'mic spectrum.pdf',
    'th.pdf'
]


dpgram_expected_suffixes = [
    'dpgram.csv',
    'dpgram.pdf',
    'mic spectrum.pdf',
]


def _process_file(fh, cb):
    '''
    Given file containing DPOAE data, recalculate DPOAE from mic signal
    '''
    fs = fh.system_microphone.fs
    ramp_time = fh.get_setting('primary_tone_rise_time')
    window = fh.get_setting('response_window')

    n_window = window * fs
    n_trim = (ramp_time * 4) * fs
    if int(n_window) != n_window:
        raise ValueError('n_window is not an integer')
    if int(n_trim) != n_trim:
        raise ValueError('n_trim is not an integer')
    n_trim = int(n_trim)
    n_window = int(n_window)
    resolution = fs / n_window

    step = 25e-3
    n_step = int(step * fs)

    cal = fh.system_microphone.get_calibration()
    psd = {}
    measured = {}
    f2_prev = None
    for i, (row, s) in enumerate(fh.iter_segments()):
        cb(i / len(fh.results))
        f2 = row['f2_frequency']
        f1 = row['f1_frequency']
        l2 = row['f2_level']
        dp = 2 * f1 - f2
        nf_freq = np.array([-2, -1, 1, 2]) * resolution + dp
        s = s.values[n_trim:]

        m_set = []
        p_set = []
        for i in range(1):
            n_segments, n_left = divmod(s.shape[-1], n_window)
            if n_left != 0:
                s = s[:-n_left]
            s_segmented = s.reshape((n_segments, -1))
            m = np.isfinite(s_segmented).all(axis=1)
            s_segmented = s_segmented[m]
            p = cal.get_db(util.psd_df(s_segmented.mean(axis=0), fs))
            s = s[n_step:]

            p_set.append(p)
            m_set.append({
                'f1_level': p[f1],
                'f2_level': p[f2],
                'dp_level': p[dp],
                'dp_nf': p[nf_freq].mean(),
                'online_dp_level': row['meas_dpoae_level'],
                'online_dp_nf': row['dpoae_noise_floor'],
            })
        measured[f2, l2] = pd.DataFrame(m_set).mean(axis=0)
        psd[f2, l2] = pd.DataFrame(p_set).mean(axis=0)

    measured = pd.DataFrame(
        measured.values(),
        index=pd.MultiIndex.from_tuples(measured.keys(), names=['f2', 'l2'])
    )
    return psd, measured


def process_file_dpoae(filename, manager):
    with manager.create_cb() as cb:
        fh = DPOAEFile(filename)
        freq = fh.results['f2_frequency'].unique()
        level = fh.results['f2_level'].unique()
        n_freq = len(freq)
        n_level = len(level)
        f2_f1_ratio = fh.get_setting('f2_f1_ratio')

        psd, measured = _process_file(fh, cb)
        io_figure, axes = plt.subplots(1, n_freq, figsize=(4 * n_freq, 4),
                                       sharex=True, sharey=True, squeeze=False)
        for fi, f2 in enumerate(freq):
            col = axes[:, fi]
            m = measured.loc[f2]

            ax = col[0]
            ax.axhline(0, ls='-', color='k')
            ax.plot(m['f2_level'], marker='o', color='0.5')
            ax.plot(m['f1_level'], marker='o', color='k')
            ax.plot(m['dp_level'], marker='o', color='darkturquoise')
            ax.plot(m['dp_nf'], marker='x', color='lightblue')
            ax.set_title(f'{f2} Hz')
            ax.set_xlabel('F2 level (dB SPL)')
            ax.grid()

        for ax in axes[:, 0]:
            ax.set_ylabel('Measured level (dB SPL)')

        mic_figure, axes = plt.subplots(n_level, n_freq,
                                        figsize=(4 * n_freq, 4 * n_level),
                                        sharex=True, sharey=True,
                                        squeeze=False)
        for fi, f2 in enumerate(freq):
            for li, l2 in enumerate(level[::-1]):
                ax = axes[li, fi]
                f1 = f2 / f2_f1_ratio
                dp = 2 * f1 - f2
                ax.axvline(f2, lw=2, color='lightblue')
                ax.axvline(f1, lw=2, color='lightblue')
                ax.axvline(dp, lw=2, color='darkturquoise')
                ax.axhline(0, lw=2, color='0.5')
                try:
                    ax.plot(psd[f2, l2].iloc[1:], color='k')
                except KeyError:
                    pass

        min_freq = min(2 * (freq / f2_f1_ratio) - freq)
        max_freq = max(freq)
        axes[0, 0].axis(xmin=min_freq * 0.8, xmax=max_freq / 0.8)
        axes[0, 0].set_xscale('octave')

        for ax in axes[-1]:
            ax.set_xlabel('Frequency (kHz)')
        for ax in axes[:, 0]:
            ax.set_ylabel('PSD (dB SPL)')

        th = measured.groupby('f2').apply(isodp_th_criterions)
        th_figure, ax = plt.subplots(1, 1, figsize=(4, 4))
        for c, row in th.T.iterrows():
            ax.plot(row, 'o-', label=str(c))
        ax.set_xscale('octave', octaves=0.5)
        ax.set_xlabel('$f_2$ frequency (kHz)')
        ax.set_ylabel('IsoDP threshold (dB SPL)')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax.axis(xmin=5.6e3*0.8, xmax=45.3e3/0.8, ymin=0, ymax=80)

        manager.save_dataframe(measured, 'io.csv')
        manager.save_dataframe(th, 'th.csv')
        manager.save_fig(mic_figure, 'mic spectrum.pdf')
        manager.save_fig(io_figure, 'io.pdf')
        manager.save_fig(th_figure, 'th.pdf')
        plt.close('all')

    return True


def process_file_dpgram(filename, manager):
    with manager.create_cb() as cb:
        fh = DPOAEFile(filename)
        freq = fh.results['f2_frequency'].unique()
        level = fh.results['f2_level'].unique()
        n_freq = len(freq)
        n_level = len(level)
        f2_f1_ratio = fh.get_setting('f2_f1_ratio')
        psd, measured = _process_file(fh, cb)
        measured = measured.reset_index().set_index(['l2', 'f2']).sort_index()

        dpgram_figure, axes = plt.subplots(1, n_level,
                                           figsize=(4 * n_level, 4),
                                           sharex=True, sharey=True,
                                           squeeze=False)
        for li, l2 in enumerate(level):
            col = axes[:, li]
            m = measured.loc[l2]

            ax = col[0]
            ax.axhline(0, ls='-', color='k')
            ax.plot(m['f2_level'], marker='o', color='0.5')
            ax.plot(m['f1_level'], marker='o', color='k')
            ax.plot(m['dp_level'], marker='o', color='darkturquoise')
            ax.plot(m['dp_nf'], marker='x', color='lightblue')
            ax.set_title(f'{l2} dB SPL')
            ax.set_xlabel('F2 frequency (kHz)')
            ax.set_xscale('octave')
            ax.grid()

        for ax in axes[:, 0]:
            ax.set_ylabel('Measured level (dB SPL)')

        mic_figure, axes = plt.subplots(n_level, n_freq,
                                        figsize=(4 * n_freq, 4 * n_level),
                                        sharex=True, sharey=True,
                                        squeeze=False)
        for fi, f2 in enumerate(freq):
            for li, l2 in enumerate(level[::-1]):
                ax = axes[li, fi]
                f1 = f2 / f2_f1_ratio
                dp = 2 * f1 - f2
                ax.axvline(f2, lw=2, color='lightblue')
                ax.axvline(f1, lw=2, color='lightblue')
                ax.axvline(dp, lw=2, color='darkturquoise')
                ax.axhline(0, lw=2, color='0.5')
                try:
                    ax.plot(psd[f2, l2].iloc[1:], color='k')
                except KeyError:
                    pass

        min_freq = min(2 * (freq / f2_f1_ratio) - freq)
        max_freq = max(freq)
        axes[0, 0].axis(xmin=min_freq * 0.8, xmax=max_freq / 0.8)
        axes[0, 0].set_xscale('octave')

        for ax in axes[-1]:
            ax.set_xlabel('Frequency (kHz)')
        for ax in axes[:, 0]:
            ax.set_xlabel('PSD (dB)')

        th = measured.groupby('f2').apply(isodp_th_criterions)
        th_figure, ax = plt.subplots(1, 1, figsize=(4, 4))
        for c, row in th.T.iterrows():
            ax.plot(row, 'o-', label=str(c))
        ax.set_xscale('octave', octaves=0.5)
        ax.set_xlabel('$f_2$ frequency (kHz)')
        ax.set_ylabel('IsoDP threshold (dB )')
        ax.legend()
        ax.axis(xmin=5.6e3*0.8, xmax=45.3e3/0.8, ymin=0, ymax=80)

        manager.save_dataframe(measured, 'dpgram.csv')
        manager.save_fig(dpgram_figure, 'dpgram.pdf')
        manager.save_fig(mic_figure, 'mic spectrum.pdf')
        plt.close('all')


def main_dpoae():
    import argparse
    parser = argparse.ArgumentParser('Summarize DPOAE IO data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    # This will exclude the dual_dpoae_io files.
    process_files('**/*[!_]dpoae_io*', process_file_dpoae,
                  expected_suffixes=dpoae_expected_suffixes, **args)


def main_dpgram():
    import argparse
    parser = argparse.ArgumentParser('Summarize DPgram data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files('**/*dpgram*', process_file_dpgram,
                  expected_suffixes=dpgram_expected_suffixes, **args)
