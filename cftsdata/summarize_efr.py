import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psiaudio import util

from .efr import EFR
from .util import add_default_options, DatasetManager, process_files


expected_suffixes = [
    'EEG bootstrapped.csv',
    'EFR.csv',
    'EFR.pdf',
    'spectrum.pdf',
    'stimulus levels.csv',
    'EFR harmonics.csv',
]


def process_file(filename, cb='tqdm', reprocess=False, segment_duration=0.5,
                 n_draw=10, n_bootstrap=100, efr_harmonics=5):
    '''
    Parameters
    ----------
    segment_duration : float
        Duration of segments to segment data into. This applies to both
        continuous (Shaheen) and epoched (Verhulst, Bramhall) approaches.
    efr_harmoincs : int
        Number of harmonics (including fundamental) to include when calculating
        EFR power.
    '''
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed(expected_suffixes):
        return

    with manager.create_cb(cb) as cb:
        fh = EFR(filename)
        n_segments = fh.get_setting('duration') / segment_duration
        if n_segments != int(n_segments):
            raise ValueError(f'Cannot analyze {filename} using default settings')
        n_segments = int(n_segments)

        mic_grouped = fh.get_mic_epochs().dropna().groupby(['fm', 'fc'])
        eeg_grouped = fh.get_eeg_epochs().dropna().groupby(['fm', 'fc'])
        cal = fh.system_microphone.get_calibration()

        keys = []
        eeg_bs_all = []
        levels_all = []


        if fh.efr_type == 'ram':
            level_harmonics = np.arange(-10, 11)
        else:
            level_harmonics = np.arange(-1, 2)

        spectrum_figures = []
        n = len(eeg_grouped)
        for i, ((fm, fc), eeg) in enumerate(eeg_grouped):
            figure, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 12))

            mic = mic_grouped.get_group((fm, fc))
            n = len(mic) * n_segments
            mic = mic.values.reshape((n, -1))
            mic_psd = util.psd_df(mic, fs=fh.mic.fs, window='hann').mean(axis=0)
            mic_spl = cal.get_db(mic_psd)
            axes[0, 0].plot(mic_spl, color='k')
            axes[0, 0].axhline(fh.level, color='forestgreen', label='Requested level')

            levels = mic_spl[fc + fm * level_harmonics]
            total_level = 10 * np.log10(np.sum(10**(levels / 10)))
            levels = levels.to_dict()
            levels['total'] = total_level
            levels_all.append(levels)

            # Plot the EEG PSD
            n = len(eeg) * n_segments
            eeg = eeg.values.reshape((n, -1))
            eeg_psd = util.db(util.psd_df(eeg, fs=fh.eeg.fs, window='hann').mean(axis=0))
            axes[0, 1].plot(eeg_psd, color='k')

            eeg_bs = util.psd_bootstrap_loop(eeg, fs=fh.eeg.fs, n_draw=n_draw, n_bootstrap=n_bootstrap)
            eeg_bs_all.append(eeg_bs)
            keys.append((fm, fc))

            axes[1, 0].plot(eeg_bs['psd_norm'], color='k')
            axes[1, 1].plot(eeg_bs['plv'], color='k')

            for ax in axes.flat:
                for i in range(1, 6):
                    ls = ':' if i != 1 else '-'
                    ax.axvline(60 * i, color='lightgray', ls=ls, zorder=-1, label='60 Hz and harmonics')
                    ax.axvline(fm * i, color='lightblue', ls=ls, zorder=-1, label='$F_m$ and harmonics')
                ax.axvline(fc, color='pink', zorder=-1, label='$F_c$ and sidebands')
                ax.axvline(fc+fm, color='pink', zorder=-1)
                ax.axvline(fc-fm, color='pink', zorder=-1)

            axes[0, 1].set_xscale('octave')
            axes[0, 1].axis(xmin=50, xmax=50e3)

            for ax in axes[-1]:
                ax.set_xlabel('Frequency (kHz)')

            axes[0, 0].set_title('Microphone')
            axes[0, 1].set_title('EEG')
            axes[1, 0].set_title('EEG (bootstrapped)')
            axes[1, 1].set_title('EEG (bootstrapped)')
            axes[0, 0].set_ylabel('Stimulus (dB SPL)')
            axes[0, 1].set_ylabel('Response (dB re 1Vrms)')
            axes[1, 0].set_ylabel('Norm. amplitude (dB re noise floor)')
            axes[1, 1].set_ylabel('Phase-locking value')

            figure.suptitle(f'{fc} Hz modulated @ {fm} Hz')
            spectrum_figures.append(figure)
            cb((i + 1) / n)

        eeg_bs_all = pd.concat(eeg_bs_all, keys=keys, names=['fm', 'fc'])
        index = pd.MultiIndex.from_tuples(keys, names=['fm', 'fc'])
        levels_all = pd.DataFrame(levels_all, index=index)
        levels_all.columns.name = 'frequency'
        levels_all = levels_all.stack().rename('level (dB SPL)')

        harmonic_power = []
        for (fm, fc), df in eeg_bs_all.groupby(['fm', 'fc']):
            # Harmonics includes the fundamental (i.e., fm)
            harmonics = np.arange(1, efr_harmonics + 1) * fm
            ix = pd.IndexSlice[:, :, harmonics]
            p = df.loc[ix].copy()

            # fm should be 0 in this array
            p.loc[:, 'harmonic'] = np.arange(efr_harmonics)
            p = p.set_index('harmonic', append=True)
            harmonic_power.append(p)

        harmonic_power = pd.concat(harmonic_power, axis=0).reset_index()
        efr = harmonic_power.query('harmonic == 0').drop(['frequency', 'harmonic'], axis='columns').set_index(['fc', 'fm'])
        efr['psd_norm_harmonics'] = util.db(harmonic_power.groupby(['fc', 'fm'])['psd_norm_linear'].sum())

        efr_figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for fm, efr_df in efr.reset_index().groupby('fm'):
            p, = axes[0].plot(efr_df['fc'], efr_df['psd'], 'o-', label=f'{fm} Hz')
            c = p.get_color()
            axes[1].plot(efr_df['fc'], efr_df['psd_norm'], 'o:', label=f'{fm} Hz ($f_0$)', color=c)
            axes[1].plot(efr_df['fc'], efr_df['psd_norm_harmonics'], 'o-', color=c, label=f'{fm} Hz ($f_{{0-20}})$')
            axes[2].plot(efr_df['fc'], efr_df['plv'], 'o-', color=c, label=f'{fm} Hz')

        axes[1].legend()
        axes[2].legend()
        axes[0].set_xscale('octave')
        for ax in axes:
            ax.set_xlabel('Carrier Freq. (kHz)')
        axes[0].set_ylabel('EFR (dB re 1V)')
        axes[1].set_ylabel('EFR (dB re noise floor)')
        axes[2].set_ylabel('Phase-locking value (frac.)')
        axes[2].axis(ymin=0, ymax=1.1)
        efr_figure.tight_layout()

        manager.save_df(harmonic_power, 'EFR harmonics.csv', index=False)
        manager.save_df(eeg_bs_all, 'EEG bootstrapped.csv')
        manager.save_df(levels_all, 'stimulus levels.csv')
        manager.save_df(efr, 'EFR.csv')
        manager.save_fig(efr_figure, 'EFR.pdf')
        manager.save_figs(spectrum_figures, 'spectrum.pdf')


def main_file():
    import argparse
    parser = argparse.ArgumentParser('Summarize EFR file')
    parser.add_argument('filename')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    process_file(args.filename, reprocess=args.reprocess)


def main_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize EFR in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*efr_ram_epoch*',
                  process_file, reprocess=args.reprocess,
                  halt_on_error=args.halt_on_error)
    process_files(args.folder, '**/*efr_sam_epoch*',
                  process_file, reprocess=args.reprocess,
                  halt_on_error=args.halt_on_error)
