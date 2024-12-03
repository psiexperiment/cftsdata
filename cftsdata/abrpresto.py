import logging
log = logging.getLogger(__name__)

import warnings

import argparse
import datetime as dt
from functools import partial
from math import ceil

from ABRpresto._version import __version__ as ABRpresto_version
from ABRpresto import XCsub

import cftsdata
from . import abr
from .util import add_default_options, process_files


COLUMNS = ['frequency', 'level', 'polarity']


expected_suffixes = [
    'ABRpresto diagnostics.pdf',
    'ABRpresto threshold.json',
    'ABRpresto settings.json',
]


XCsubargs = {
    'seed': 0,
    'pst_range': [0.0005, 0.006],
    'N_shuffles': 500,
    'avmode': 'median',
    'peak_lag_threshold': 0.5,
    'XC0m_threshold': 0.3,
    'XC0m_sigbounds': 'increasing, midpoint within one step of x range',  # sets bounds to make slope positive,
    # and midpoint within [min(level) - step, max(level) + step] step is usually 5 dB
    'XC0m_plbounds': 'increasing',  # sets bounds to make slope positive
    'second_filter': 'pre-average',
    'calc_XC0m_only': True,
    'save_data_resamples': False  # use this to save intermediate data (from each resample)
}


def process_file(filename, manager, tlb=0, tub=6e-3, target_fs=12.5e3, base_filename=''):
    '''
    Extract ABR epochs, filter and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR experiment. If it's a set of ABR experiments, epochs across
        all experiments will be combined for the analysis.
    manager : instance of DatasetManager
        DatasetManager for handling data storage and review.
    target_fs : float
        TODO
    '''
    settings = locals()
    settings.pop('manager')
    settings.pop('filename')
    settings['creation_time'] = dt.datetime.now().isoformat()
    settings['versions'] = {
        'cftsdata': cftsdata.__version__,
        'ABRpresto': ABRpresto_version,
    }

    with manager.create_cb():
        fh = abr.load(filename)
        downsample = int(ceil(fh.eeg.fs / target_fs))
        settings['downsample'] = downsample
        settings['actual_fs'] = fh.eeg.fs / downsample
        epochs = fh.get_epochs_filtered(downsample=downsample,
                                        columns=['frequency', 'level', 'polarity'])

        result = {}
        figures = []
        for freq, freq_df in epochs.groupby('frequency'):
            r, fig = XCsub.estimate_threshold(freq_df, **XCsubargs)
            result[freq] = r
            fig.suptitle(abr.freq_to_label(freq))
            figures.append(fig)

        manager.save_dict(result, 'ABRpresto threshold.json')
        manager.save_dict(settings, 'ABRpresto settings.json')
        manager.save_figs(figures, 'ABRpresto diagnostics.pdf')


def main():
    parser = argparse.ArgumentParser('Threshold ABR data using ABRpresto')
    add_default_options(parser)
    args = vars(parser.parse_args())

    fn = partial(process_file)
    process_files(glob_pattern='**/*abr_io*', fn=fn, expected_suffixes=expected_suffixes, **args)
