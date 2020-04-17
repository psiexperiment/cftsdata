import argparse
import datetime as dt
from glob import glob
import json
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from psi.data.io import abr


columns = ['frequency', 'level', 'polarity']


def process_folder(folder, filter_settings=None):
    glob_pattern = os.path.join(folder, '*abr*')
    filenames = glob(glob_pattern)
    process_files(filenames, filter_settings=filter_settings)


def process_files(filenames, offset=-0.001, duration=0.01,
                  filter_settings=None, reprocess=False):
    for filename in filenames:
        try:
            processed = process_file(filename, offset, duration,
                                     filter_settings, reprocess)
            if processed:
                print(f'\nProcessed {filename}\n')
            else:
                print('*', end='', flush=True)
        except Exception as e:
            raise
            print(f'\nError processing {filename}\n{e}\n')


def _get_file_template(fh, offset, duration, filter_settings, suffix=None,
                       simple_filename=True):
    if simple_filename:
        return 'ABR {}.csv'

    base_string = f'ABR {offset*1e3:.1f}ms to {(offset+duration)*1e3:.1f}ms'
    if filter_settings == 'saved':
        settings = _get_filter(fh)
        if not settings['digital_filter']:
            filter_string = None
        else:
            lb = settings['lb']
            ub = settings['ub']
            filter_string = f'{lb:.0f}Hz to {ub:.0f}Hz filter'
    elif filter_settings is None:
        filter_string = None
    else:
        lb = filter_settings['lb']
        ub = filter_settings['ub']
        filter_string = f'{lb:.0f}Hz to {ub:.0f}Hz filter'
        order = filter_settings['order']
        if order != 1:
            filter_string = f'{order:.0f} order {filter_string}'

    if filter_string is None:
        file_string = f'{base_string}'
    else:
        file_string = f'{base_string} with {filter_string}'

    if suffix is not None:
        file_string = f'{file_string} {suffix}'

    return f'{file_string} {{}}.csv'


def _get_filter(fh):
    if not isinstance(fh, (abr.ABRFile, abr.ABRSupersetFile)):
        fh = abr.load(fh)
    return {
        'digital_filter': fh.get_setting_default('digital_filter', True),
        'lb': fh.get_setting_default('digital_highpass', 300),
        'ub': fh.get_setting_default('digital_lowpass', 3000),
        # Filter order is not currently an option in the psiexperiment ABR
        # program so it defaults to 1.
        'order': 1,
    }

def _get_epochs(fh, offset, duration, filter_settings, reject_ratio=None):
    # We need to do the rejects in this code so that we can obtain the
    # information for generating the CSV files. Set reject_threshold to np.inf
    # to ensure that nothing gets rejected.
    kwargs = {'offset': offset, 'duration': duration, 'columns': columns,
              'reject_threshold': np.inf}
    if filter_settings is None:
        return fh.get_epochs(**kwargs)

    if filter_settings == 'saved':
        settings = _get_filter(fh)
        if not settings['digital_filter']:
            return fh.get_epochs(**kwargs)
        lb = settings['lb']
        ub = settings['ub']
        order = settings['order']
        kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
        return fh.get_epochs_filtered(**kwargs)
    lb = filter_settings['lb']
    ub = filter_settings['ub']
    order = filter_settings['order']
    kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
    return fh.get_epochs_filtered(**kwargs)


def _match_epochs(*epochs):

    def _match_n(df):
        grouping = df.groupby(['dataset', 'polarity'])
        n = grouping.size().unstack()
        if len(n) < 2:
            return None
        n = n.values.ravel().min()
        return pd.concat([g.iloc[:n] for _, g in grouping])

    epochs = pd.concat(epochs, keys=range(len(epochs)), names=['dataset'])
    matched = epochs.groupby(['frequency', 'level']).apply(_match_n)
    return [d.reset_index('dataset', drop=True) for _, d in \
              matched.groupby('dataset', group_keys=False)]


def is_processed(filename, offset, duration, filter_settings, suffix=None,
                 simple_filename=True):
    t = _get_file_template(filename, offset, duration, filter_settings, suffix,
                           simple_filename)
    file_template = os.path.join(filename, t)
    raw_epoch_file = file_template.format('individual waveforms')
    mean_epoch_file = file_template.format('average waveforms')
    n_epoch_file = file_template.format('number of epochs')
    return os.path.exists(raw_epoch_file) and \
        os.path.exists(mean_epoch_file) and \
        os.path.exists(n_epoch_file)


def process_file(filename, offset, duration, filter_settings, reprocess=False,
                 n_epochs='auto', suffix=None, simple_filename=True):
    '''
    Extract ABR epochs, filter and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR experiment. If it's a set of ABR experiments, epochs across
        all experiments will be combined for the analysis.
    offset : sec
        The start of the epoch to extract, in seconds, relative to tone pip
        onset. Negative values can be used to extract a prestimulus baseline.
    duration: sec
        The duration of the epoch to extract, in seconds, relative to the
        offset. If offset is set to -0.001 sec and duration is set to 0.01 sec,
        then the epoch will be extracted from -0.001 to 0.009 sec re tone pip
        onset.
    filter_settings : {None, 'saved', dict}
        If None, no additional filtering is done. If 'saved', uses the digital
        filter settings that were saved in the ABR file. If a dictionary, must
        contain 'lb' (the lower bound of the passband in Hz) and 'ub' (the
        upper bound of the passband in Hz).
    reprocess : bool
        If True, reprocess the file even if it already has been processed for
        the specified filter settings.
    n_epochs : {None, 'auto', int, dict}
        If None, all epochs will be used. If 'auto', use the value defined at
        acquisition time. If integer, will limit the number of epochs per
        frequency and level to this number. If dict, the key must be a tuple of
        (frequency, level) and the value will indicate the number of epochs to
        use.
    suffix : {None, str}
        Suffix to use when creating save filenames.
    simple_filename : bool
        Pass
    '''
    settings = locals()
    settings['filename'] = str(settings['filename'])
    settings['creation_time'] = dt.datetime.now().isoformat()
    fh = abr.load(filename)
    if len(fh.erp_metadata) == 0:
        raise IOError('No data in file')

    t = _get_file_template(fh, offset, duration, filter_settings, suffix,
                           simple_filename)
    file_template = os.path.join(filename, t)
    raw_epoch_file = file_template.format('individual waveforms')
    mean_epoch_file = file_template.format('average waveforms')
    n_epoch_file = file_template.format('number of epochs')
    reject_ratio_file = file_template.format('reject ratio')
    settings_file = Path(file_template.format('processing settings')).with_suffix('.json')

    # Check to see if all of them exist before reprocessing
    if not reprocess and \
            (os.path.exists(raw_epoch_file) and \
             os.path.exists(mean_epoch_file) and \
             os.path.exists(n_epoch_file) and \
             os.path.exists(reject_ratio_file)):
        return False

    # Load the epochs
    epochs = _get_epochs(fh, offset, duration, filter_settings)

    # Apply the reject
    reject_threshold = fh.get_setting('reject_threshold')
    m = np.abs(epochs) < reject_threshold
    m = m.all(axis=1)
    epochs = epochs.loc[m]

    if n_epochs is not None:
        if n_epochs == 'auto':
            n_epochs = fh.get_setting('averages')
        n = int(np.floor(n_epochs / 2))
        epochs = epochs.groupby(columns) \
            .apply(lambda x: x.iloc[:n])

    epoch_reject_ratio = 1-m.groupby(columns[:-1]).mean()
    epoch_mean = epochs.groupby(columns).mean() \
        .groupby(columns[:-1]).mean()

    # Write the data to CSV files
    epoch_reject_ratio.name = 'epoch_reject_ratio'
    epoch_reject_ratio.to_csv(reject_ratio_file, header=True)
    epoch_reject_ratio.name = 'epoch_n'
    epoch_n = epochs.groupby(columns[:-1]).size()
    epoch_n.to_csv(n_epoch_file, header=True)
    epoch_mean.columns.name = 'time'
    epoch_mean.T.to_csv(mean_epoch_file)
    epochs.columns.name = 'time'
    epochs.T.to_csv(raw_epoch_file)
    settings_file.write_text(json.dumps(settings, indent=2))
    return True


def main_auto():
    parser = argparse.ArgumentParser('Filter and summarize ABR files in folder')
    parser.add_argument('folder', type=str, help='Folder containing ABR data')
    args = parser.parse_args()
    process_folder(args.folder, filter_settings='saved')


def main():
    parser = argparse.ArgumentParser('Filter and summarize ABR data')

    parser.add_argument('filenames', type=str,
                        help='Filename', nargs='+')
    parser.add_argument('--offset', type=float,
                        help='Epoch offset',
                        default=-0.001)
    parser.add_argument('--duration', type=float,
                        help='Epoch duration',
                        default=0.01)
    parser.add_argument('--filter-lb', type=float,
                        help='Highpass filter cutoff',
                        default=None)
    parser.add_argument('--filter-ub', type=float,
                        help='Lowpass filter cutoff',
                        default=None)
    parser.add_argument('--order', type=float,
                        help='Filter order',
                        default=None)
    parser.add_argument('--reprocess',
                        help='Redo existing results',
                        action='store_true')
    args = parser.parse_args()

    if args.filter_lb is not None or args.filter_ub is not None:
        filter_settings = {
            'lb': args.filter_lb,
            'ub': args.filter_ub,
            'order': args.order,
        }
    else:
        filter_settings = None
    process_files(args.filenames, args.offset, args.duration, filter_settings,
                  args.reprocess)


def main_gui():
    import enaml
    from enaml.qt.qt_application import QtApplication
    with enaml.imports():
        from .summarize_abr_gui import SummarizeABRGui

    app = QtApplication()
    view = SummarizeABRGui()
    view.show()
    app.start()


if __name__ == '__main__':
    main_gui()
