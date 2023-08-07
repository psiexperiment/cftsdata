import datetime as dt
import json
import re
import os
from pathlib import Path

import pandas as pd

from cftsdata.abr import load_abr_analysis


P_PSI_FILENAME = re.compile(
	'^(?P<datetime>\d{8}-\d{6}) '
	'(?P<experimenter>\w+) '
	'(?P<animal_id>B?\d+-(\d+|C)) '
	'(?P<ear>left|right) (?P<note>.*) '
	'(?P<experiment_type>(?:abr|dpoae|efr|memr)_\w+).*$'
)


def parse_psi_filename(filename):
    try:
        groups = P_PSI_FILENAME.match(filename.stem).groupdict()
        groups['datetime'] = dt.datetime.strptime(groups['datetime'], '%Y%m%d-%H%M%S')
        groups['date'] = groups['datetime'].date()
        groups['time'] = groups['datetime'].time()
        return groups
    except AttributeError:
        raise ValueError(f'Could not parse {filename.stem}')


class Dataset:

    def __init__(self, ephys_path=None, subpath=None):
        if ephys_path is None:
            ephys_path = os.environ['PROC_DATA_DIR']
        self.ephys_path = Path(ephys_path)
        if subpath is not None:
            self.ephys_path = self.ephys_path / subpath

    def _load(self, cb, glob, filename_parser, data_path=None):
        if data_path is None:
            data_path = self.ephys_path
        result = []
        for filename in data_path.glob(glob):
            if '_exclude' in str(filename):
                continue
            if '.imaris_cache' in str(filename):
                continue
            data = cb(filename)
            for k, v in filename_parser(filename).items():
                if k in data:
                    raise ValueError('Column will get overwritten')
                data[k] = v
            result.append(data)
        if len(result) == 0:
            raise ValueError('No data found')
        if isinstance(data, pd.DataFrame):
            df = pd.concat(result)
        else:
            df = pd.DataFrame(result)
        return df

    def load_dpoae_io(self):
        return self._load(lambda x: pd.read_csv(x),
                          '**/*dpoae_io io.csv',
                          parse_psi_filename)

    def load_dpoae_th(self):
        def _load_dpoae_th(x):
            df = pd.read_csv(x, index_col=0)
            df.columns = df.columns.astype('f')
            df.columns.name = 'criterion'
            return df.stack().rename('threshold').reset_index()
        return self._load(_load_dpoae_th,
                          '**/*dpoae_io th.csv',
                          parse_psi_filename)

    def load_abr_io(self):
        def _load_abr_io(x):
            freq, th, rater, peaks = load_abr_analysis(x)
            peaks = peaks.reset_index()
            peaks['frequency'] = freq
            peaks['rater'] = rater
            return peaks
        abr_io = self._load(_load_abr_io,
                               '**/*analyzed.txt',
                               parse_psi_filename)
        abr_io['w1'] = abr_io.eval('p1_amplitude - n1_amplitude')
        return abr_io

    def load_abr_th(self):
        def _load_abr_th(x):
            freq, th, rater, _ = load_abr_analysis(x)
            return pd.Series({'frequency': freq, 'threshold': th, 'rater': rater})
        return self._load(_load_abr_th,
                          '**/*analyzed.txt',
                          parse_psi_filename)

    def load_abr_settings(self):
        def _load_abr_settings(x):
            return pd.Series(json.loads(x.read_text()))
        return self._load(_load_abr_settings,
                          '**/*ABR experiment settings.json',
                          parse_psi_filename)

    def load_abr_eeg_spectrum(self):
        def _load_abr_eeg_spectrum(x):
            df = pd.read_csv(x, index_col=0)
            df.columns = ['psd']
            return df
        return self._load(_load_abr_eeg_spectrum,
                          '**/*ABR eeg spectrum.csv',
                          parse_psi_filename)

    def load_abr_eeg_rms(self):
        def _load_abr_eeg_rms(x):
            return pd.Series(json.loads(x.read_text()))
        return self._load(_load_abr_eeg_rms,
                          '**/*ABR eeg rms.json',
                          parse_psi_filename)
