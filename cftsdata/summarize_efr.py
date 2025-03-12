import datetime as dt

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
    'EFR harmonics.csv',
    'stimulus levels.csv',
    'stimulus SPL.csv',
    'stimulus SPL.pdf',
    'EEG spectrum.pdf',
    'EFR processing settings.json',
]


def extract_nf(psd, frequencies, n_bins=4, n_sep=3):
    freq = psd.index.get_level_values('frequency')
    freq_step = freq[1] - freq[0]
    # Indices of the noise bins relative to the bin containing the harmonic.
    # Delete the center value (0) so we don't include the harmonic in the
    # calculation of the noise floor.
    lb = np.arange(-n_bins-n_sep, -n_sep)
    ub = np.arange(n_sep+1, n_sep+n_bins+1)
    i_noise = np.concatenate((lb, ub))
    result = {}
    for f in frequencies:
        i_h = int(f // freq_step)
        i_nf = i_h + i_noise
        result[f] = psd.iloc[i_nf].mean()
    return pd.Series(result)


def plot_eeg(eeg_bs_all, harmonics):
    all_fm = sorted(eeg_bs_all.index.unique('fm'))
    all_fc = sorted(eeg_bs_all.index.unique('fc'))
    n_fm = len(all_fm)
    n_fc = len(all_fc)

    figure, axes = plt.subplots(n_fm, n_fc, figsize=(4*n_fc, 4*n_fm),
                                sharex='row', sharey='row', squeeze=False)

    for fm, row_df in eeg_bs_all.groupby('fm'):
        for fc, df in row_df.groupby('fc'):
            ax = axes[all_fm.index(fm), all_fc.index(fc)]
            x = df['psd'].reset_index(['fm', 'fc'], drop=True).loc[1:fm*6]
            ax.plot(x)

            h = harmonics.query('(fm == @fm) & (fc == @fc)')
            ax.plot(h['frequency'], h['psd'], 'ko')
            ax.plot(h['frequency'], h['psd_nf_bins'], '_', color='darkred', ms=10)

            for i in range(1, 6):
                ax.axvline(i * fm, lw=3, alpha=0.1, color='k', zorder=0)

        axes[all_fm.index(fm), 0].axis(xmin=0, xmax=6*fm)

    for i, fc in enumerate(all_fc):
        axes[0, i].set_title(f'$f_c$ {fc*1e-3:.1f} kHz')
    for i, fm in enumerate(all_fm):
        axes[i, 0].set_ylabel(f'$f_m$ {fm:.0f} Hz\nPSD (dB re 1V)')

    for ax in axes[-1]:
        ax.set_xlabel('Frequency (Hz)')

    return figure


def plot_spl(spl, requested_level, actual_level):
    all_fm = sorted(spl.index.unique('fm'))
    all_fc = sorted(spl.index.unique('fc'))
    n_fm = len(all_fm)
    n_fc = len(all_fc)

    figure, axes = plt.subplots(n_fm, n_fc, figsize=(4*n_fc, 4*n_fm),
                                sharex='row', sharey='row', squeeze=False)

    for fm, row_df in spl.groupby('fm'):
        for fc, df in row_df.groupby('fc'):
            ax = axes[all_fm.index(fm), all_fc.index(fc)]
            x = df.reset_index(['fm', 'fc'], drop=True).loc[1:]
            ax.plot(x)
            ax.axhline(requested_level, color='forestgreen', label='Requested level')
            ax.axhline(actual_level.loc[fm, fc], color='salmon', label='Actual level')

        axes[all_fm.index(fm), 0].set_xscale('octave')
        axes[all_fm.index(fm), 0].axis(xmin=5.6e3, xmax=45.2e3)

    for i, fc in enumerate(all_fc):
        axes[0, i].set_title(f'$f_c$ {fc*1e-3:.1f} kHz')
    for i, fm in enumerate(all_fm):
        axes[i, 0].set_ylabel(f'$f_m$ {fm:.0f} Hz\nPSD (dB re 1V)')

    for ax in axes[-1]:
        ax.set_xlabel('Frequency (Hz)')

    return figure


def extract_harmonics(eeg_bs, n_harmonics):
    harmonic_power = []
    for (fm, fc), df in eeg_bs.groupby(['fm', 'fc']):
        harmonics = np.arange(1, n_harmonics+1) * fm
        ix = pd.IndexSlice[:, :, harmonics]
        p = df.loc[ix].copy()

        p.loc[:, 'harmonic'] = np.arange(n_harmonics)
        p = p.set_index('harmonic', append=True)

        nf = extract_nf(df['psd'], harmonics)
        nf.name = 'psd_nf_bins'
        p = p.join(nf, on='frequency')
        harmonic_power.append(p)

    return pd.concat(harmonic_power, axis=0).reset_index()


def bootstrap_eeg(eeg_df, n_draw, n_bootstrap, cb=None):
    eeg_bs = {}
    eeg_grouped = eeg_df.groupby(['fm', 'fc'])
    n = len(eeg_grouped)
    for i, (key, df) in enumerate(eeg_grouped):
        n_pol = len(df.index.get_level_values('polarity').unique())
        if n_pol == 2:
            arrays = [
                df.xs(-1, level='polarity').values,
                df.xs(1, level='polarity').values,
            ]
        elif n_pol == 1:
            # Oldest experiments only had a single polarity.
            arrays = df.values
        else:
            raise ValueError('Unexpected polarity')

        eeg_bs[key] = util.psd_bootstrap_loop(arrays,
                                                  fs=eeg_df.attrs['fs'],
                                                  n_draw=n_draw,
                                                  n_bootstrap=n_bootstrap,
                                                  callback=None,
                                                  window='tukey')
        if cb is not None:
            cb((i + 1) / n)
    return pd.concat(eeg_bs, names=['fm', 'fc'])


def get_spl(mic_df, cal):
    spl = {}
    for key, df in mic_df.groupby(['fm', 'fc']):
        df_mean = df.groupby('polarity').mean()
        df_mean = 0.5 * (df_mean.loc[1] - df_mean.loc[-1])
        psd = util.psd_df(df_mean, fs=mic_df.attrs['fs'], window='hann')
        spl[key] = cal.get_db(psd)
    return pd.concat(spl, names=['fm', 'fc']).rename('SPL')


def get_level(spl, efr_type):
    level_harmonics = np.arange(-10, 11) if efr_type == 'ram' else np.arange(-1, 2)

    level_spl = {}
    for (fm, fc), df in spl.groupby(['fm', 'fc'], group_keys=False):
        df = df.reset_index(['fm', 'fc'], drop=True)
        level_freqs = fc + fm * level_harmonics
        max_freq = df.index.max()
        level_freqs = level_freqs[(level_freqs > 0) & (level_freqs <= max_freq)]
        level_spl[fm, fc] = df.loc[level_freqs]

    return pd.concat(level_spl, names=['fm', 'fc']).rename('SPL')


def process_file(filename, manager, segment_duration=0.5, n_draw=128,
                 n_bootstrap=100, efr_harmonics=5, target_fs=12500):
    '''
    Parameters
    ----------
    segment_duration : float
        Duration of segments to segment data into. This applies to both
        continuous (Shaheen) and epoched (Verhulst, Bramhall) approaches.
    efr_harmonics : int
        Number of harmonics (including fundamental) to include when calculating
        EFR power. If None, calculate up to the maximum available.
    target_fs : float
        Target sampling rate to decimate EEG data to. Downsampling greatly
        speeds up the bootstrap analyses. Be sure the target sampling rate is
        at least twice the maximum harmonic you want to analyze in the EFR data.
    '''
    # Force a copy since the pandas query later does something very, very odd
    # to the dictionary returned. Probably a bug? It seems that the pandas
    # query causes the dictionary returned by `locals` to be modified in-place
    # with new variables.
    settings = dict(locals())
    settings.pop('manager')
    settings['filename'] = str(settings['filename'])
    settings['creation_time'] = dt.datetime.now().isoformat()
    settings['n_harmonics'] = efr_harmonics

    with manager.create_cb() as cb:
        fh = EFR(filename)

        eeg_df = fh.get_eeg_epochs(target_fs=target_fs,
                                   segment_duration=segment_duration,
                                   columns=['fm', 'fc', 'polarity']).dropna()
        eeg_bs = bootstrap_eeg(eeg_df, n_draw, n_bootstrap, cb)
        harmonics = extract_harmonics(eeg_bs, efr_harmonics)
        spectrum_figure = plot_eeg(eeg_bs, harmonics)

        mic_df = fh.get_mic_epochs(segment_duration=segment_duration)
        cal = fh.system_microphone.get_calibration()
        spl = get_spl(mic_df, cal)
        level = get_level(spl, fh.efr_type)
        total_level = level.groupby(['fm', 'fc']).apply(lambda x: 10 * np.log10(np.sum(10**(x / 10))))
        spl_figure = plot_spl(spl, fh.level, total_level)

        manager.save_df(eeg_bs, 'EEG bootstrapped.csv')
        manager.save_df(harmonics, 'EFR harmonics.csv', index=False)
        manager.save_df(spl, 'stimulus spl.csv')
        manager.save_df(level, 'stimulus levels.csv')
        manager.save_fig(spectrum_figure, 'EEG spectrum.pdf')
        manager.save_fig(spl_figure, 'stimulus SPL.pdf')
        manager.save_dict(settings, 'EFR processing settings.json')


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize EFR in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*efr_ram*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
    process_files(glob_pattern='**/*efr_sam*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
