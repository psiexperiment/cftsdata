import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import util

from .efr import EFR
from .util import add_default_options, process_files

from psiaudio.efr import efr_bs_verhulst


expected_suffixes = [
    'EEG bootstrapped.csv',
    'EFR.csv',
    'EFR.pdf',
    'spectrum.pdf',
    'stimulus levels.csv',
    'EFR harmonics.csv',

    # Verhulst method
    'EFR amplitude linear.csv',
    'EFR harmonics linear.csv',
    'EFR PSD linear.csv',
]


def process_file(filename, manager, segment_duration=0.5, n_draw=128,
                 n_bootstrap=100, efr_harmonics=5, target_fs=12500):
    '''
    Parameters
    ----------
    segment_duration : float
        Duration of segments to segment data into. This applies to both
        continuous (Shaheen) and epoched (Verhulst, Bramhall) approaches.
    efr_harmoincs : int
        Number of harmonics (including fundamental) to include when calculating
        EFR power.
    target_fs : float
        Target sampling rate to decimate EEG data to. Downsampling greatly
        speeds up the bootstrap analyses. Be sure the target sampling rate is
        at least twice the maximum harmonic you want to analyze in the EFR data.
    '''
    with manager.create_cb() as cb:
        fh = EFR(filename)
        n_segments = fh.get_setting('duration') / segment_duration
        if n_segments != int(n_segments):
            raise ValueError(f'Cannot analyze {filename} using default settings')
        n_segments = int(n_segments)

        # Calculate the decimation factor and the actual sampling rate of the
        # downsampled EEG data.
        n_dec = int(fh.eeg.fs // target_fs)
        actual_fs = fh.eeg.fs / n_dec

        mic_grouped = fh.get_mic_epochs(columns=['fm', 'fc', 'polarity']).dropna().groupby(['fm', 'fc'])
        eeg_raw = fh.get_eeg_epochs(columns=['fm', 'fc', 'polarity']).dropna()
        eeg_dec = pd.DataFrame(
            signal.decimate(eeg_raw, n_dec),
            index=eeg_raw.index,
            columns=eeg_raw.columns[::n_dec],
        )
        eeg_grouped = eeg_dec.groupby(['fm', 'fc'])

        cal = fh.system_microphone.get_calibration()

        keys = []
        eeg_bs_all = []
        levels_all = []
        v_amplitude_all = []
        v_harmonics_all = []
        v_psd_all = []

        if fh.efr_type == 'ram':
            level_harmonics = np.arange(-10, 11)
        else:
            level_harmonics = np.arange(-1, 2)

        spectrum_figures = []
        n = len(eeg_grouped)
        for i, ((fm, fc), eeg_df) in enumerate(eeg_grouped):
            figure, axes = plt.subplots(3, 2, sharex=False, figsize=(12, 18),
                                        layout='constrained')

            mic = mic_grouped.get_group((fm, fc))
            n = len(mic) * n_segments
            mic = mic.values.reshape((n, -1))
            mic_psd = util.psd_df(mic, fs=fh.mic.fs, window='hann').mean(axis=0)
            mic_spl = cal.get_db(mic_psd)
            axes[0, 0].plot(mic_spl, color='k')
            axes[0, 0].axhline(fh.level, color='forestgreen', label='Requested level')

            # Remove extra frequencies (i.e., DC and negative frequencies) and
            # then calculate total level.
            level_freqs = fc + fm * level_harmonics
            level_freqs = level_freqs[(level_freqs > 0) & (level_freqs <= (fh.mic.fs / 2))]

            levels = util.tone_power_conv(mic, fh.mic.fs, level_freqs).mean(axis=-1)
            levels = cal.get_db(level_freqs, levels)
            total_level = 10 * np.log10(np.sum(10**(levels / 10)))
            levels = {f: l for f, l in zip(level_freqs, levels)}
            levels['total'] = total_level
            levels_all.append(levels)

            axes[0, 0].axhline(total_level, color='salmon', label='Measured level')

            # Plot the EEG PSD
            n = len(eeg_df) * n_segments
            eeg = eeg_df.values.reshape((n, -1))
            eeg_psd = util.db(util.psd_df(eeg.mean(axis=0), fs=actual_fs, window='hann'))
            axes[0, 1].plot(eeg_psd, color='k')

            eeg_bs = util.psd_bootstrap_loop(eeg, fs=actual_fs, n_draw=n_draw,
                                             n_bootstrap=n_bootstrap,
                                             callback=None)
            eeg_bs_all.append(eeg_bs)
            keys.append((fm, fc))

            axes[1, 0].plot(eeg_bs['psd_norm'], color='k')
            axes[1, 1].plot(eeg_bs['plv'], color='k')

            for ax in axes.flat:
                for i in range(1, 6):
                    ax.axvline(60 * i, color='lightgray', ls=':', zorder=-1, label='60 Hz and harmonics')
                    ax.axvline(fm * i, color='lightblue', ls='-', zorder=-1, label='$F_m$ and harmonics')
                ax.axvline(fc, color='pink', zorder=-1, label='$F_c$')

            for ax in axes.flat:
                ax.set_xscale('octave')
                ax.set_xlabel('Frequency (kHz)')
                ax.axis(xmin=50, xmax=6e3)

            axes[0, 0].axis(xmin=50, xmax=50e3)
            axes[0, 0].set_title(f'Microphone ({total_level:.2f} dB SPL)')
            axes[0, 1].set_title('EEG')
            axes[1, 0].set_title('EEG (bootstrapped)')
            axes[1, 1].set_title('EEG (bootstrapped)')
            axes[0, 0].set_ylabel('Stimulus (dB SPL)')
            axes[0, 1].set_ylabel('Response (dB re 1Vrms)')
            axes[1, 0].set_ylabel('Norm. amplitude (dB re noise floor)')
            axes[1, 1].set_ylabel('Phase-locking value')

            # Plot the second 10 cycles of the filtered waveform (first 10
            # cycles may have onset artifact).
            f_lb = fm / 1.2
            f_ub = fm * efr_harmonics * 1.2
            b, a = signal.iirfilter(2, (f_lb, f_ub), btype='band',
                                    ftype='butter', fs=actual_fs)

            eeg_filt = signal.filtfilt(b, a, eeg, axis=-1).mean(axis=0)
            t = np.arange(len(eeg_filt)) / actual_fs
            axes[2, 0].plot(t * 1e3, eeg_filt)
            axes[2, 0].axis(xmin=10/fm * 1e3, xmax=20/fm * 1e3)
            axes[2, 0].set_title(f'Raw waveform\nFiltered from {f_lb*1e-3:.1f} to {f_ub*1e-3:.1f} kHz')
            axes[2, 0].set_xscale('linear')
            axes[2, 0].set_xlabel('Time (msec)')
            axes[2, 0].set_ylabel('Amplitude (V)')

            # Now, calculate bootstrapped EFR using Verhulst approach and plot
            # diagnostics.
            v_amplitude, v_harmonics, v_psd = efr_bs_verhulst(eeg,
                                                              fs=actual_fs,
                                                              n_draw=n_draw,
                                                              n_bootstrap=n_bootstrap,
                                                              fm=fm,
                                                              n_harmonics=efr_harmonics)
            v_amplitude_all.append(v_amplitude)
            v_harmonics_all.append(v_harmonics)
            v_psd_all.append(v_psd)

            # Plot the calculated EFR amplitude
            a = v_amplitude.mean() * np.sqrt(2)
            axes[2, 0].axhline(a, color='salmon')
            axes[2, 0].axhline(-a, color='salmon')

            # Plot the bootstrapped PSD (unnormalized)
            axes[2, 1].plot(util.db(v_psd.mean(axis=0)).iloc[1:])

            # Plot the calculated amplitude and noise floor of each harmonic.
            v_mean_harmonics = v_harmonics.groupby('harmonic').mean()
            for h, h_row in v_mean_harmonics.iterrows():
                f_lb, f_ub = h / 1.1, h * 1.1
                a = util.db(h_row['amplitude'])
                nf = util.db(h_row['noise_floor'])
                axes[2, 1].plot([h], [a], 'o', mec='salmon', mfc='none')
                axes[2, 1].plot([f_lb, f_ub], [nf, nf], '-', color='salmon')

            axes[2, 1].set_title('EFR harmonics')
            axes[2, 1].set_xlabel('Frequency (kHz)')
            axes[2, 1].set_ylabel('Amplitude (dB re 1V)')

            figure.suptitle(f'{fc} Hz modulated @ {fm} Hz')
            spectrum_figures.append(figure)
            cb((i + 1) / n)

        eeg_bs_all = pd.concat(eeg_bs_all, keys=keys, names=['fm', 'fc'])
        v_amplitude_all = pd.concat(v_amplitude_all, keys=keys, names=['fm', 'fc'])
        v_harmonics_all = pd.concat(v_harmonics_all, keys=keys, names=['fm', 'fc'])
        v_psd_all = pd.concat(v_psd_all, keys=keys, names=['fm', 'fc'])

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
        efr['amplitude'] = v_amplitude_all.groupby(['fc', 'fm']).mean()
        efr['amplitude_db'] = util.db(efr['amplitude'])

        efr_figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for fm, efr_df in efr.reset_index().groupby('fm'):
            p, = axes[0].plot(efr_df['fc'], efr_df['amplitude_db'], 'o-', label=f'{fm} Hz')
            c = p.get_color()
            axes[1].plot(efr_df['fc'], efr_df['psd_norm'], 'o:',
                         label=f'{fm} Hz ($f_0$)', color=c)
            axes[1].plot(efr_df['fc'], efr_df['psd_norm_harmonics'], 'o-', color=c,
                         label=f'{fm} Hz ($f_{{0-{efr_harmonics-1}}})$')
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
        manager.save_df(v_amplitude_all, 'EFR amplitude linear.csv')
        manager.save_df(v_harmonics_all, 'EFR harmonics linear.csv')
        manager.save_df(v_psd_all, 'EFR PSD linear.csv')
        manager.save_fig(efr_figure, 'EFR.pdf')
        manager.save_figs(spectrum_figures, 'spectrum.pdf')

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize EFR in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*efr_ram*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
    process_files(glob_pattern='**/*efr_sam*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
