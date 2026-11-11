import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import util
from psidata.manager import add_default_options, process_files

from .efr import EFR
from .loaders.efr import SUFFIXES

import cftsdata
import psiaudio
# Registers the 'octave' axis scale
import psiaudio.plot
import psidata


expected_suffixes = [
    SUFFIXES['psd'],
    SUFFIXES['harmonics'],
    'EEG spectrum.pdf',
    SUFFIXES['processing_settings'],
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
    settings['version'] = {
        'psiaudio': psiaudio.__version__,
        'cftsdata': cftsdata.__version__,
    }

    with manager.create_cb() as cb:
        fh = EFR(filename)

        # Process and save the EEG
        eeg_df = fh.get_eeg_epochs(target_fs=target_fs,
                                   segment_duration=segment_duration,
                                   columns=['fm', 'fc', 'polarity']).dropna()
        eeg_bs = bootstrap_eeg(eeg_df, n_draw, n_bootstrap, cb)
        harmonics = extract_harmonics(eeg_bs, efr_harmonics)
        spectrum_figure = plot_eeg(eeg_bs, harmonics)
        manager.save_df(eeg_bs, 'EEG bootstrapped.csv')
        manager.save_df(harmonics, 'EFR harmonics.csv', index=False)
        manager.save_fig(spectrum_figure, 'EEG spectrum.pdf')
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


# Separate entrypoint for checking the stimulus for distortion artifacts

EXPECTED_SUFFIXES_SPL = [
    SUFFIXES['mic_spectrum'],
    SUFFIXES['mic_levels'],
    'mic spectrum.pdf',
    SUFFIXES['mic_processing_settings'],
]


# Settings used by `process_file_spl`. Saved to the processing settings file.
MIC_SETTINGS = {
    # The averaged waveform is split into segments of this duration and the
    # power of the segments averaged (i.e., Welch's method). The frequency
    # resolution is 1 / segment_duration.
    'segment_duration': 0.5,
    'window': 'hann',
    'detrend': 'constant',
    # Number of harmonics of fm to mark in the plots.
    'n_fm_harmonics': 5,
    # Sidebands (fc + k * fm) summed to give the actual stimulus level.
    'level_sidebands': {'sam': [-1, 0, 1], 'ram': list(range(-10, 11))},
}

# Component of the microphone signal containing the stimulus
STIMULUS_COMPONENTS = ('difference', 'single')

COMPONENT_LABELS = {
    'difference': '(pos - neg) / 2: stimulus + odd-order distortion',
    'sum': '(pos + neg) / 2: even-order distortion',
    'single': 'single polarity',
}


def find_microphones(fh):
    '''
    Names of all microphone signals saved in the recording

    The name depends on the version of the CFTS launcher and the hardware
    (e.g., `system_microphone`, `starship_A_microphone_monitor`,
    `microphone_monitor_monitor`).
    '''
    # psidata does not offer a public way to list the signals.
    signals = set(fh._store.zarr_names) | set(fh._store.carray_names)
    return sorted(s for s in signals if 'microphone' in s)


def calibrated_range(cal):
    '''
    Range of frequencies (Hz) covered by the calibration, None if all
    '''
    frequency = getattr(cal, 'frequency', None)
    if frequency is None:
        return None
    return [float(frequency.min()), float(frequency.max())]


def get_db_calibrated(cal, psd):
    '''
    Convert `psd` to dB SPL, NaN at frequencies outside the calibrated range
    '''
    cal_range = calibrated_range(cal)
    if cal_range is None:
        return cal.get_db(psd)
    lb, ub = cal_range
    m = (psd.index >= lb) & (psd.index <= ub)
    return cal.get_db(psd[m]).reindex(psd.index)


def get_requested_level(fh, md):
    '''
    Requested stimulus level (dB SPL) from the metadata for one stimulus
    '''
    column = 'target_tone_level' if fh.efr_type == 'ram' else 'target_sam_tone_level'
    if column not in md:
        return fh.level
    levels = md[column].unique()
    if len(levels) != 1:
        raise ValueError(f'Multiple requested levels {levels} for one stimulus')
    return float(levels[0])


def get_stimulus_level(waveform, fs, cal, fm, fc, sidebands):
    '''
    Level (dB SPL) of the carrier and sidebands, fc + k * fm for k in sidebands

    Sidebands above Nyquist or outside the calibrated range are skipped.
    '''
    k = np.asarray(sidebands)
    frequency = fc + fm * k
    m = (frequency > 0) & (frequency <= fs / 2)
    cal_range = calibrated_range(cal)
    if cal_range is not None:
        m &= (frequency >= cal_range[0]) & (frequency <= cal_range[1])
    k, frequency = k[m], frequency[m]
    rms = util.tone_power_conv(waveform, fs, frequency)
    return pd.DataFrame({
        'sideband': k,
        'frequency': frequency,
        'SPL': cal.get_db(frequency, rms),
    })


def get_mic_spectrum(fh, mic, segment_duration, window, detrend, sidebands,
                     cb=None):
    '''
    Spectrum and level of the averaged stimulus recorded by the microphone

    Epochs are averaged within polarity (time-locked to stimulus onset), then
    combined across polarity (see `cftsdata.loaders.efr.mic_spectrum`). The
    result is split into segments and the power of the segments averaged.

    Returns
    -------
    spl : pd.DataFrame
        Level (dB SPL) indexed by fm, fc and component, one column per
        frequency.
    levels : pd.DataFrame
        Level of the carrier and each sideband in `sidebands` measured from the
        stimulus component, with the requested level, one row per sideband
        (see `cftsdata.loaders.efr.mic_levels`).
    n_epochs : dict
        Number of epochs averaged, keyed by (fm, fc, polarity).
    '''
    md = fh.analyze_efr_metadata
    duration = fh.get_setting('duration')
    fs = mic.fs
    cal = mic.get_calibration()
    n_segment = int(round(segment_duration * fs))

    spl = {}
    levels = {}
    n_epochs = {}
    grouped = md.groupby(['fm', 'fc'])
    for i, ((fm, fc), md_group) in enumerate(grouped):
        # Load one stimulus at a time to limit memory use.
        epochs = mic.get_epochs(md_group, 0, duration, detrend=detrend,
                                columns=['polarity']).dropna()
        epoch_mean = epochs.groupby('polarity').mean()
        for p, n in epochs.groupby('polarity').size().items():
            n_epochs[fm, fc, p] = n

        polarities = sorted(epoch_mean.index)
        if polarities == [-1, 1]:
            pos, neg = epoch_mean.loc[1], epoch_mean.loc[-1]
            components = {'difference': 0.5 * (pos - neg),
                          'sum': 0.5 * (pos + neg)}
        elif len(polarities) == 1:
            components = {'single': epoch_mean.iloc[0]}
        else:
            raise ValueError(f'Unexpected polarities {polarities}')

        for name, waveform in components.items():
            n = len(waveform) // n_segment
            if n == 0:
                raise ValueError(f'Stimulus duration {duration} s is shorter '
                                 f'than the segment duration {segment_duration} s')
            segments = waveform.values[:n * n_segment].reshape((n, n_segment))
            psd = util.psd_df(pd.DataFrame(segments), fs=fs, window=window)
            # Average power across segments
            psd = np.sqrt((psd ** 2).mean(axis=0))
            spl[fm, fc, name] = get_db_calibrated(cal, psd)

            if name in STIMULUS_COMPONENTS:
                level = get_stimulus_level(waveform.values, fs, cal, fm, fc,
                                           sidebands)
                level.insert(0, 'requested_level', get_requested_level(fh, md_group))
                levels[fm, fc] = level

        if cb is not None:
            cb((i + 1) / len(grouped))

    spl = pd.DataFrame(spl).T
    spl.index.names = ['fm', 'fc', 'component']
    spl.columns.name = 'frequency'
    levels = pd.concat(levels, names=['fm', 'fc']) \
        .reset_index(level=-1, drop=True).reset_index()
    return spl, levels, n_epochs


def total_level(levels):
    '''
    Sum the power across sidebands to give the actual level of each stimulus

    Parameters
    ----------
    levels : pd.DataFrame
        As returned by `get_mic_spectrum` (with an optional `channel` column).

    Returns
    -------
    df : pd.DataFrame
        `requested_level`, `actual_level` and `difference` (actual - requested)
        for each stimulus.
    '''
    grouping = [c for c in ('channel', 'fm', 'fc') if c in levels]
    result = levels.groupby(grouping).agg(
        requested_level=('requested_level', 'first'),
        actual_level=('SPL', lambda x: 10 * np.log10(np.sum(10 ** (x / 10)))),
    )
    result['difference'] = result['actual_level'] - result['requested_level']
    return result


def plot_mic_spectrum(spl, level, channel, n_fm_harmonics):
    '''
    Plot the full spectrum and a zoom on the low frequencies around fm

    One row per stimulus (fm, fc). The requested and actual level (see
    `total_level`) are marked on the full spectrum.
    '''
    groups = list(spl.groupby(['fm', 'fc']))
    figure, axes = plt.subplots(len(groups), 2, figsize=(12, 3.5 * len(groups)),
                                squeeze=False, constrained_layout=True)
    nyquist = spl.columns.max()
    for (ax_full, ax_zoom), ((fm, fc), df) in zip(axes, groups):
        df = df.droplevel(['fm', 'fc'])
        for component, row in df.iterrows():
            label = COMPONENT_LABELS[component]
            # Components are labeled in the legend on the zoom panel
            ax_full.plot(row.loc[1:], lw=0.5)
            ax_zoom.plot(row.loc[:(n_fm_harmonics + 1) * fm], label=label)
        harmonics = fm * np.arange(1, n_fm_harmonics + 1)
        for ax in (ax_full, ax_zoom):
            for h in harmonics:
                ax.axvline(h, color='k', alpha=0.1, lw=3, zorder=0)
        for h in fc * np.arange(1, 4):
            if h < nyquist:
                ax_full.axvline(h, color='darkred', alpha=0.2, lw=3, zorder=0)
        requested, actual = level.loc[(fm, fc), ['requested_level', 'actual_level']]
        ax_full.axhline(requested, color='forestgreen', ls='--',
                        label=f'Requested level ({requested:.1f} dB SPL)')
        ax_full.axhline(actual, color='salmon', ls='--',
                        label=f'Actual level ({actual:.1f} dB SPL)')
        ax_full.legend(loc='lower left', fontsize=8)
        ax_full.set_xscale('octave')
        ax_full.axis(xmin=fm / 1.2, xmax=nyquist)
        ax_full.set_xlabel('Frequency (kHz)')
        ax_full.set_title(f'$f_m$ {fm:.0f} Hz, $f_c$ {fc * 1e-3:.2f} kHz')
        ax_zoom.axis(xmin=0, xmax=(n_fm_harmonics + 1) * fm)
        ax_zoom.set_xlabel('Frequency (Hz)')
        ax_zoom.set_title(f'Harmonics of $f_m$ (1-{n_fm_harmonics})')
        ax_full.set_ylabel('Level (dB SPL)')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    figure.suptitle(f'{channel}\nGray: harmonics of $f_m$. Red: $f_c$, 2$f_c$, 3$f_c$')
    return figure


def process_file_spl(filename, manager):
    with manager.create_cb() as cb:
        fh = EFR(filename)
        channels = find_microphones(fh)
        if not channels:
            raise ValueError('No microphone signal saved in recording')

        md = fh.analyze_efr_metadata
        settings = {
            'filename': str(filename),
            'creation_time': dt.datetime.now().isoformat(),
            'version': {
                'cftsdata': cftsdata.__version__,
                'psiaudio': psiaudio.__version__,
                'psidata': getattr(psidata, '__version__', None),
            },
            **MIC_SETTINGS,
            'stimulus_duration': fh.get_setting('duration'),
            'stimulus_system': md['system'].unique().tolist() if 'system' in md else None,
            'components': COMPONENT_LABELS,
            'channels': {},
        }

        sidebands = MIC_SETTINGS['level_sidebands'][fh.efr_type]
        spectra = {}
        levels = {}
        figures = []
        for i, channel in enumerate(channels):
            def channel_cb(frac, i=i):
                cb((i + frac) / len(channels))
            mic = getattr(fh, channel)
            spl, level, n_epochs = get_mic_spectrum(
                fh, mic, MIC_SETTINGS['segment_duration'],
                MIC_SETTINGS['window'], MIC_SETTINGS['detrend'], sidebands,
                cb=channel_cb)
            spectra[channel] = spl
            levels[channel] = level
            figures.append(plot_mic_spectrum(spl, total_level(level), channel,
                                             MIC_SETTINGS['n_fm_harmonics']))
            cal = mic.get_calibration()
            settings['channels'][channel] = {
                'fs': mic.fs,
                'calibration': type(cal).__name__,
                # Levels outside this range are NaN in the spectrum
                'calibrated_range': calibrated_range(cal),
                'n_epochs': [
                    {'fm': fm, 'fc': fc, 'polarity': p, 'n': n}
                    for (fm, fc, p), n in n_epochs.items()
                ],
            }

        spectra = pd.concat(spectra, names=['channel'])
        levels = pd.concat(levels, names=['channel']) \
            .reset_index(level=-1, drop=True).reset_index()
        manager.save_df(spectra, SUFFIXES['mic_spectrum'])
        manager.save_df(levels, SUFFIXES['mic_levels'], index=False)
        manager.save_figs(figures, 'mic spectrum.pdf')
        manager.save_dict(settings, SUFFIXES['mic_processing_settings'])
        for figure in figures:
            plt.close(figure)


def main_spl():
    import argparse
    parser = argparse.ArgumentParser('Summarize EFR microphone spectrum in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*efr_ram*', fn=process_file_spl,
                  expected_suffixes=EXPECTED_SUFFIXES_SPL, **args)
    process_files(glob_pattern='**/*efr_sam*', fn=process_file_spl,
                  expected_suffixes=EXPECTED_SUFFIXES_SPL, **args)
