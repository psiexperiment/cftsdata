from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import util
from psidata.api import Recording

from .util import add_default_options, DatasetManager, process_files


expected_suffixes = [
    'ECG.pdf',
    'BPM.csv',
    'instantaneous BPM.csv',
]


def process_file(filename, manager, plot_seconds=5, bpm_window=15, bpm_step=1):
    '''
    Parameters
    ----------
    '''
    with manager.create_cb() as cb:
        fh = Recording(filename)
        eeg = fh.eeg[0]
        cb(0.4)
        fs = fh.eeg.fs

        n_plot = int(round(fs * plot_seconds))

        b, a = signal.iirfilter(2, (5, 400), fs=fs)
        eeg_filtered = signal.filtfilt(b, a, eeg)
        cb(0.8)

        rms = util.rms(eeg_filtered)
        th = rms * 4
        min_distance = int(fs * 10e-3)

        peaks, _ = signal.find_peaks(np.abs(eeg_filtered), distance=min_distance, height=th)

        figure = plt.figure(figsize=(10, 7.5), constrained_layout=True)
        gs = plt.GridSpec(3, 3, figure=figure)
        ax_eeg = figure.add_subplot(gs[0, :2])
        ax_eeg_final = figure.add_subplot(gs[1, :2], sharey=ax_eeg)
        ax_hr = figure.add_subplot(gs[2, :])
        ax_ecg = figure.add_subplot(gs[0, 2], sharey=ax_eeg)

        t = np.arange(len(eeg_filtered)) / fs

        p_subset = peaks[peaks < n_plot]
        ax_eeg.plot(t[:n_plot], eeg[:n_plot], lw=1, color='0.5', alpha=0.5)
        ax_eeg.axhline(+th, color='coral')
        ax_eeg.axhline(-th, color='coral')
        ax_eeg.plot(t[:n_plot], eeg_filtered[:n_plot], lw=0.5, color='k')
        ax_eeg.plot(t[p_subset], eeg[p_subset], 'o', color='seagreen', alpha=0.5)
        ax_eeg.set_xlabel('Time (sec)')
        ax_eeg.set_ylabel('Amplitude (V)')
        ax_eeg.axis(xmin=0, xmax=5, ymin=-10, ymax=10)

        p_subset = peaks[peaks >= (len(eeg_filtered) - n_plot)]
        ax_eeg_final.plot(t[-n_plot:], eeg[-n_plot:], lw=1, color='0.5', alpha=0.5)
        ax_eeg_final.axhline(+th, color='coral')
        ax_eeg_final.axhline(-th, color='coral')
        ax_eeg_final.plot(t[-n_plot:], eeg_filtered[-n_plot:], lw=0.5, color='k')
        ax_eeg_final.plot(t[p_subset], eeg[p_subset], 'o', color='seagreen', alpha=0.5)
        ax_eeg_final.set_xlabel('Time (sec)')
        ax_eeg_final.set_ylabel('Amplitude (V)')
        ax_eeg_final.axis(xmin=t[p_subset].min(), xmax=t[p_subset].max())

        # Calculate heartrate using sliding window
        ecg = np.zeros(len(eeg))
        ecg[peaks] = 1
        n_window = int(round(bpm_window * fs))
        n_step = int(round(bpm_step * fs))
        n_ecg = np.lib.stride_tricks.sliding_window_view(ecg, n_window)[::n_step].sum(axis=1)
        bpm = n_ecg / bpm_window * 60
        t_bpm = np.arange(len(bpm)) * bpm_step + bpm_window * 0.5

        # Calculate instantaneous heartrate (this often generates outliers)
        irate = fs / np.diff(peaks) * 60
        t_irate = peaks[1:] / fs
        ax_hr.plot(t_irate, irate, '-', color='0.5', alpha=0.5)
        ax_hr.plot(t_bpm, bpm, 'k-')
        ax_hr.axis(ymin=0, ymax=600)
        ax_hr.grid()
        ax_hr.set_xlabel('Time (sec)')
        ax_hr.set_ylabel('Heartrate (BPM)')

        lb, ub = -5e-3, 10e-3
        lbi, ubi = int(round(lb*fs)), int(round(ub*fs))

        ecg = []
        for p in peaks:
            if ((p + lbi) < 0) or ((p + ubi) > len(eeg_filtered)):
                continue
            ecg.append(eeg_filtered[p+lbi:p+ubi])
        ecg = np.vstack(ecg)

        t_ecg = np.arange(ecg.shape[-1]) / fs + lb
        ax_ecg.plot(t_ecg * 1e3, ecg.T, 'k-', lw=0.1, alpha=0.1)
        ax_ecg.set_xlabel('Time re. ECG peak (msec)')

        hr = pd.DataFrame({'time': t_irate, 'instantaneous_heartrate': irate})
        manager.save_df(hr, 'instantaneous BPM.csv', index=False)
        hr = pd.DataFrame({'time': t_bpm, 'heartrate': bpm})
        manager.save_df(hr, 'BPM.csv', index=False)
        manager.save_fig(figure, 'ECG.pdf')


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize ECG data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*abr_io*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
    process_files(glob_pattern='**/*efr_ram*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
    process_files(glob_pattern='**/*efr_sam*', fn=process_file,
                  expected_suffixes=expected_suffixes, **args)
