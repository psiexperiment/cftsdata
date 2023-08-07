from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from psidata.api import Recording
from psiaudio import util

from .util import add_default_options, DatasetManager, process_files


expected_suffixes = [
    'noise exposure.json',
    'noise exposure.pdf',
    'SPL over time.csv',
    'PSD.csv',
]


def process_file(filename, cb='tqdm', reprocess=False, start_delay=1,
                 analysis_window=10, segment_duration=60):
    '''
    {STANDARD_PARAMS}
    start_delay : float
        Delay of noise exposure onset, in seconds.
    analysis_window : float
        Window for analyzing noise RMS and FFT over.
    segment_duration : float
        Maximum segment to load from disk at a time. These arrays are fairly
        large, so don't set it to a very large number or you'll run out of
        memory.
    '''
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed(expected_suffixes):
        return

    with manager.create_cb(cb) as cb:
        fh = Recording(filename)
        parameters = fh.get_parameters()
        noise_fh = float(parameters['exposure_bandlimited_noise_fh'])
        noise_fl = float(parameters['exposure_bandlimited_noise_fl'])

        noise_level = parameters['exposure_bandlimited_noise_level']
        if '+' in noise_level:
            noise_level, noise_correction = noise_level.split('+')
            noise_correction = float(noise_correction)
        elif '-' in noise_level:
            noise_level, noise_correction = noise_level.split('-')
            noise_correction = -float(noise_correction)
        else:
            noise_correction = 0
        noise_level = float(noise_level)

        n_samples = fh.monitor_microphone.shape[-1]
        fs = fh.monitor_microphone.fs
        cal = fh.monitor_microphone.get_calibration()

        n_analysis_samples = int(round(analysis_window * fs))
        n_segment_samples = int(round(segment_duration * fs))
        # Ensure segment duration is an integer multiple of the analysis window
        n_segment_samples = int(round(n_segment_samples / n_analysis_samples)) * n_analysis_samples

        i_start = int(round(start_delay * fs))
        n_segments = (n_samples - i_start) // n_segment_samples

        b, a = signal.iirfilter(2, 100, None, None, 'highpass', ftype='butter', fs=fs)

        mic_mean = []
        mic_rms = []
        mic_psd = []

        for i in range(n_segments):
            cb(i/n_segments)
            lb = i_start + n_segment_samples * i
            ub = lb + n_segment_samples
            mic = fh.monitor_microphone[0, lb:ub].reshape((-1, n_analysis_samples))
            mic_mean.append(mic.mean(axis=1))
            mic_detrend = signal.detrend(mic, -1, 'linear')
            mic_detrend = signal.filtfilt(b, a, mic_detrend)
            mic_rms.append(util.rms(mic_detrend, axis=-1))
            mic_psd.append(util.psd_df(mic_detrend, fs=fs))

        mic_psd = pd.concat(mic_psd, axis=0, ignore_index=True)
        mic_psd.index.name = 'time'
        mic_spl = cal.get_spl(mic_psd.mean(axis=0))
        mic_mean = np.concatenate(mic_mean, axis=-1)
        mic_rms = np.concatenate(mic_rms, axis=-1)

        mic_rms_spl = cal.get_spl(1, mic_rms)
        time = np.arange(len(mic_rms_spl)) * analysis_window + analysis_window
        index = pd.Index(time, name='time')
        mic_psd.index = index
        mic_rms_spl = pd.Series(mic_rms_spl, index=index, name='SPL')
        mic_rms_spl_mean = cal.get_spl(1, np.mean(mic_rms))

        mic_band_rms = mic_psd.loc[:, noise_fl:noise_fh].apply(util.rms_rfft, axis=1)
        mic_band_spl = cal.get_spl(mic_band_rms).rename('SPL')
        mic_band_spl_mean = cal.get_spl(1, np.mean(mic_band_rms))

        n_bins = mic_psd.loc[:, noise_fl:noise_fh].shape[-1]
        noise_spectrum_level = util.band_to_spectrum_level(noise_level, n_bins)

        figure = plt.Figure(figsize=(12, 6), constrained_layout=True)
        gs = GridSpec(2, 4, figure=figure)

        ax_offset = figure.add_subplot(gs[0, :2])
        ax_time = figure.add_subplot(gs[1, :2], sharex=ax_offset)
        ax_fft = figure.add_subplot(gs[:, 2:])

        ax_offset.plot(time, mic_mean * 1e3, 'k-')
        ax_time.plot(mic_rms_spl, 'k-', label='Overall SPL')
        ax_time.plot(mic_band_spl, '-', color='skyblue', label='SPL in noise band')

        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Noise level (dB SPL)')
        ax_offset.set_ylabel('Mic offset (mV)')
        ax_time.axhline(noise_level)

        text = f'''
        Correction factor {noise_correction} dB
        Average level {mic_rms_spl_mean:.2f} dB SPL
        Average level in noise band {mic_band_spl_mean:.2f} dB SPL
        '''
        figure.suptitle(text)

        ax_fft.axhline(noise_spectrum_level)
        ax_fft.plot(mic_spl.iloc[1:], 'k-')
        ax_fft.axvspan(noise_fl, noise_fh, alpha=0.25)
        ax_fft.set_xscale('octave')
        ax_fft.axis(xmin=1e3, xmax=50e3)
        ax_fft.set_ylabel('Level (dB SPL)')
        ax_fft.set_xlabel('Frequency (kHz)')
        ax_time.legend()

        noise_info = {
            'freq_lb': noise_fl,
            'freq_ub': noise_fh,
            'requested_noise_level': noise_level,
            'correction_factor': noise_correction,
            'expected_spectrum_level': noise_spectrum_level,
            'measured_noise_level': mic_rms_spl_mean,
            'measured_noise_band_level': mic_band_spl_mean,
        }

        manager.save_dict(noise_info, 'noise exposure.json')
        manager.save_fig(figure, 'noise exposure.pdf')
        manager.save_df(mic_spl.rename('SPL'), 'PSD.csv')
        manager.save_df(mic_rms_spl, 'SPL over time.csv')


def main_file():
    import argparse
    parser = argparse.ArgumentParser('Summarize noise exposure file')
    parser.add_argument('filename')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    process_file(args.filename, reprocess=args.reprocess)


def main_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize noise exposure data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*noise_exposure*',
                  process_file, reprocess=args.reprocess,
                  halt_on_error=args.halt_on_error)
