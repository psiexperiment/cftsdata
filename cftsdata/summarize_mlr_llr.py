import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from . import abr
from psidata.manager import add_default_options, process_files
from psiaudio import plot


EXPECTED_SUFFIXES = [
    'ABR average waveforms.csv',
    'MLR average waveforms.csv',
    'LLR average waveforms.csv',
    'waveforms.pdf',
]



def load_epochs(fh):
    downsample = int(np.ceil(fh.eeg.fs / 10e3))
    epochs = fh.get_epochs(offset=-100e-3, duration=400e-3, columns=['frequency', 'level', 'polarity'], reject_threshold='saved-last', downsample=downsample)

    epochs_mean = epochs.groupby(['frequency', 'level']).mean()
    b, a = signal.butter(2, [300, 3000], 'band', fs=epochs.attrs['fs'])
    abr = epochs_mean.apply(lambda x: signal.filtfilt(b, a, x), axis=1, result_type='broadcast')
    abr = abr.loc[:, -0.001:0.01]
    b, a = signal.butter(2, [10, 100], 'band', fs=epochs.attrs['fs'])
    mlr = epochs_mean.apply(lambda x: signal.filtfilt(b, a, x), axis=1, result_type='broadcast')
    mlr = mlr.loc[:, -0.01:0.05]
    b, a = signal.butter(2, [1, 30], 'band', fs=epochs.attrs['fs'])
    llr = epochs_mean.apply(lambda x: signal.filtfilt(b, a, x), axis=1, result_type='broadcast')
    llr = llr.loc[:, -0.05:0.3]
    return {
        'ABR': abr,
        'MLR': mlr,
        'LLR': llr,
    }


def plot_epochs(epochs):
    freq = epochs['ABR'].index.unique('frequency')
    figure, axes = plt.subplots(3, len(freq), figsize=(4 * len(freq), 12), squeeze=False)

    epochs_grouped = {k: v.groupby('frequency') for k, v in epochs.items()}
    for col, f in zip(axes.T, freq):
        for ax, name in zip(col, ('ABR', 'MLR', 'LLR')):
            ax.set_xlabel('Time (s)')
            w = epochs_grouped[name].get_group(f)
            plot.waterfall_plot(ax, w)
            if name == 'MLR':
                ax.axvspan(0, 0.01, alpha=0.25)
                ax.text(0, 1, 'ABR', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=8)
            elif name == 'LLR':
                ax.axvspan(0, 0.05, alpha=0.25)
                ax.text(0, 1, 'MLR', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=8)
        col[0].set_title(str(f))
    axes[0, 0].set_ylabel('ABR')
    axes[1, 0].set_ylabel('MLR')
    axes[2, 0].set_ylabel('LLR')
    return figure


def process_file(filename, manager):
    with manager.create_cb() as cb:
        fh = abr.load(filename)
        epochs = load_epochs(fh)
        figure = plot_epochs(epochs)

        for k, v in epochs.items():
            manager.save_dataframe(v, f'{k} average waveforms.csv')
        manager.save_fig(figure, 'waveforms.pdf')


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize MLR/LLR files in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*mlr_llr_io*', fn=process_file,
                  expected_suffixes=EXPECTED_SUFFIXES, **args)
