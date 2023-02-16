import matplotlib.pyplot as plt

from psiaudio import util

from .efr import EFR
from .util import add_default_options, DatasetManager, process_files


def process_file(filename, cb='tqdm', reprocess=False, segment_duration=0.1,
                 n_draw=10, n_bootstrap=200):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed('psd.csv'):
        return
    manager.clear()
    cb = manager.create_cb(cb)
    cb(0)

    fh = EFR(filename)
    n_segments = fh.get_setting('duration') / segment_duration
    level = fh.get_setting('target_sam_tone_level')
    if n_segments != int(n_segments):
        raise ValueError(f'Cannot analyze {filename} using default settings')
    n_segments = int(n_segments)

    figure, axes = plt.subplots(3, 2, sharex=True, figsize=(20, 30))
    mic_grouped = fh.get_mic_epochs().groupby(['fm', 'fc'])
    eeg_grouped = fh.get_eeg_epochs().groupby(['fm', 'fc'])
    cal = fh.system_microphone.get_calibration()

    for (fm, fc), eeg in eeg_grouped:
        if len(eeg) != 1:
            raise ValueError('Cannot analyze {filename} using default settings')
        mic = mic_grouped.get_group((fm, fc))
        eeg = eeg.values[0].reshape((n_segments, -1))[1:-1]
        mic = mic.values[0].reshape((n_segments, -1))[1:-1]
        mic_psd = util.psd_df(mic, fs=fh.mic.fs, window='hann').mean(axis=0)
        eeg_psd = util.db(util.psd_df(eeg, fs=fh.eeg.fs, window='hann').mean(axis=0))
        mic_spl = cal.get_db(mic_psd)

        axes[0, 0].plot(mic_spl, color='k')
        axes[0, 0].axhline(level, color='lightblue')
        axes[0, 1].plot(eeg_psd, color='k')

        mic_bs = util.psd_bootstrap_loop(mic, fs=fh.mic.fs, n_draw=n_draw, n_bootstrap=n_bootstrap)
        eeg_bs = util.psd_bootstrap_loop(eeg, fs=fh.eeg.fs, n_draw=n_draw, n_bootstrap=n_bootstrap)
        axes[1, 0].plot(mic_bs['psd_norm'], color='k')
        axes[1, 1].plot(eeg_bs['psd_norm'], color='k')
        axes[2, 0].plot(mic_bs['plv'], color='k')
        axes[2, 1].plot(eeg_bs['plv'], color='k')

        for ax in axes.flat:
            for i in range(1, 5):
                ls = ':' if i != 1 else '-'
                ax.axvline(60 * i, color='lightgray', ls=ls, zorder=-1)
                ax.axvline(fm * i, color='lightblue', ls=ls, zorder=-1)
            ax.axvline(fc, color='pink', zorder=-1)
            ax.axvline(fc+fm, color='pink', zorder=-1)
            ax.axvline(fc-fm, color='pink', zorder=-1)

        axes[0, 1].set_xscale('octave')
        axes[0, 1].axis(xmin=50, xmax=50e3)

        for ax in axes[-1]:
            ax.set_xlabel('Frequency (kHz)')
        axes[0, 0].set_title('Microphone')
        axes[0, 1].set_title('EEG')
        axes[0, 0].set_ylabel('Stimulus (dB SPL)')
        axes[0, 1].set_ylabel('Response (dB re 1Vrms)')
        axes[1, 0].set_ylabel('Norm. amplitude (dB re noise floor)')
        axes[2, 0].set_ylabel('Phase-locking value')

        plt.show()
        return


def main_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize IEC data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*inear_speaker_calibration_chirp*',
                  process_file, reprocess=args.reprocess)


if __name__ == '__main__':
    #process_file(r'C:\Users\mmm\projects\psi1\data\data\20230127-124721 Sean Sean left 151122 1 efr_sam.zip')
    process_file(r'C:\Users\mmm\projects\psi1\data\data\20230208-113146 Sean B018-1 left 400uM efr_sam')
