import datetime as dt
import json
import re
import os
from pathlib import Path
import zipfile

from jsonpath_ng import parse
import pandas as pd

from cftsdata.abr import load_abr_analysis


class BadFilenameException(Exception):

    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return f'Bad filename: {self.filename}'


P_PSI_FILENAME = re.compile(
	'^(?P<datetime>\d{8}-\d{6}) '
	'(?P<experimenter>\w+) '
	'(?P<animal_id>B?\d+-(\d+|C)) '
	'((?P<ear>left|right) )?(?P<note>.*) '
	'(?P<experiment_type>(?:abr|dpoae|efr|memr|inear|dpgram|dual_dpoae)(_\w+)?).*$'
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


def load_efr_level(filename, which='total'):
    result = pd.read_csv(filename, index_col=['fm', 'fc', 'frequency']) \
        ['level (dB SPL)'].rename('level').reset_index()
    if which == 'total':
        result = result.query('frequency == "total"').drop(columns=['frequency'])
    elif which == 'harmonics':
        result = result.query('frequency != "total"')
        result['frequency'] = result['frequency'].astype('f')
    else:
        raise ValueError(f'Unrecognized parameter for which: "{which}"')
    return result


class Dataset:

    def __init__(self, ephys_path=None, subpath=None, raw_ephys_path=None):
        if ephys_path is None:
            ephys_path = os.environ['PROC_DATA_DIR']
        if raw_ephys_path is None:
            raw_ephys_path = os.environ['RAW_DATA_DIR']

        self.ephys_path = Path(ephys_path)
        self.raw_ephys_path = Path(raw_ephys_path)
        if subpath is not None:
            self.ephys_path = self.ephys_path / subpath
            self.raw_ephys_path = self.raw_ephys_path / subpath
        if not self.ephys_path.exists():
            raise ValueError(f'Unknown data path {self.ephys_path}')

    def load_raw(self, cb, etype=None, **kwargs):
        wildcard = '**/*.zip' if etype is None else f'**/*{etype}*.zip'
        return self.load(cb, wildcard, data_path=self.raw_ephys_path, **kwargs)

    def load_experiment_info(self, **kwargs):
        return self.load_raw(lambda x: {}, filename_parser=parse_psi_filename,
                             **kwargs)

    def load_raw_json(self, filename, json_path, etype=None):
        '''
        Load value from JSON file saved to raw data
        '''
        expr = parse(json_path)
        def cb(zip_filename):
            nonlocal expr
            nonlocal filename
            with zipfile.ZipFile(zip_filename) as fh_zip:
                with fh_zip.open(filename) as fh:
                    result = expr.find(json.load(fh))
                    if len(result) > 1:
                        raise ValueError(f'More than one match found for path "{json_path}"')
                    return {str(result[0].full_path): result[0].value}
        return self.load_raw(cb, etype)

    def load(self, cb, glob, filename_parser=None, data_path=None, include_dataset=False):
        if data_path is None:
            data_path = self.ephys_path
        if filename_parser is None:
            filename_parser = parse_psi_filename
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
            if include_dataset:
                data['dataset'] = filename.parent
            result.append(data)
        if len(result) == 0:
            raise ValueError('No data found')
        if isinstance(data, pd.DataFrame):
            df = pd.concat(result)
        else:
            df = pd.DataFrame(result)
        return df

    def load_dpoae_io(self, **kwargs):
        return self.load(lambda x: pd.read_csv(x),
                          '**/*dpoae_io io.csv',
                          parse_psi_filename, **kwargs)

    def load_dpoae_th(self, **kwargs):
        def _load_dpoae_th(x):
            df = pd.read_csv(x, index_col=0)
            df.columns = df.columns.astype('f')
            df.columns.name = 'criterion'
            return df.stack().rename('threshold').reset_index()
        return self.load(_load_dpoae_th,
                          '**/*dpoae_io th.csv',
                          parse_psi_filename, **kwargs)

    def load_abr_io(self, **kwargs):
        def _load_abr_io(x):
            freq, th, rater, peaks = load_abr_analysis(x)
            peaks = peaks.reset_index()
            peaks['frequency'] = freq
            peaks['rater'] = rater
            return peaks
        abr_io = self.load(_load_abr_io,
                            '**/*analyzed.txt',
                            parse_psi_filename,
                            **kwargs)
        abr_io['w1'] = abr_io.eval('p1_amplitude - n1_amplitude')
        return abr_io

    def load_abr_th(self, rater=None, **kwargs):
        def _load_abr_th(x):
            freq, th, rater, _ = load_abr_analysis(x)
            return pd.Series({'frequency': freq, 'threshold': th, 'rater': rater})
        glob = '**/*-analyzed.txt' if rater is None else f'**/*-{rater}-analyzed.txt'
        return self.load(_load_abr_th,
                          glob,
                          parse_psi_filename,
                          **kwargs)

    def load_abr_settings(self, **kwargs):
        def _load_abr_settings(x):
            return pd.Series(json.loads(x.read_text()))
        return self.load(_load_abr_settings,
                          '**/*ABR experiment settings.json',
                          parse_psi_filename,
                          **kwargs)

    def load_abr_frequencies(self, **kwargs):
        '''
        Loads ABR frequencies

        Returns
        -------
        frequencies : pd.DataFrame
            Dataframe with one row per frequency.
        '''
        def _load_abr_frequencies(x):
            with x.open() as fh:
                frequencies = set(float(v) for v in fh.readline().split(',')[1:])
                return pd.Series({'frequencies': sorted(frequencies)})
        ds = self.load(_load_abr_frequencies,
                        '**/*ABR average waveforms.csv',
                        parse_psi_filename, **kwargs)
        result = []
        for _, row in ds.iterrows():
            for frequency in row['frequencies']:
                new_row = row.copy()
                del new_row['frequencies']
                new_row['frequency'] = frequency
                result.append(new_row)
        return pd.DataFrame(result)

    def load_abr_eeg_spectrum(self, **kwargs):
        def _load_abr_eeg_spectrum(x):
            df = pd.read_csv(x, index_col=0)
            df.columns = ['psd']
            return df
        return self.load(_load_abr_eeg_spectrum,
                          '**/*ABR eeg spectrum.csv',
                          parse_psi_filename, **kwargs)

    def load_abr_eeg_rms(self, **kwargs):
        def _load_abr_eeg_rms(x):
            return pd.Series(json.loads(x.read_text()))
        return self.load(_load_abr_eeg_rms,
                          '**/*ABR eeg rms.json',
                          parse_psi_filename, **kwargs)

    def load_efr_sam_linear(self, **kwargs):
        def _load_efr_sam_linear(x):
            return pd.read_csv(x).groupby(['fc', 'fm'])['efr_amplitude'] \
                .agg(['mean', 'std']).add_prefix('efr_sam_linear_').reset_index()
        return self.load(_load_efr_sam_linear,
                         '**/*efr_sam*EFR amplitude linear.csv',
                         parse_psi_filename, **kwargs)

    def load_efr_ram_linear(self, **kwargs):
        def _load_efr_ram_linear(x):
            return pd.read_csv(x).groupby(['fc', 'fm'])['efr_amplitude'] \
                .agg(['mean', 'std']).add_prefix('efr_ram_linear_').reset_index()
        return self.load(_load_efr_ram_linear,
                         '**/*efr_ram*EFR amplitude linear.csv',
                         parse_psi_filename, **kwargs)

    def load_efr_sam_level(self, **kwargs):
        return self.load(load_efr_level,
                         '**/*efr_sam*stimulus levels.csv',
                         parse_psi_filename, **kwargs)

    def load_efr_ram_level(self, **kwargs):
        return self.load(load_efr_level,
                         '**/*efr_ram*stimulus levels.csv',
                         parse_psi_filename, **kwargs)
