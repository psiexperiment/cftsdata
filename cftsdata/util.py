import json
import os
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class CallbackManager:

    def __init__(self, cb, autoclose_figures=True):
        self._cb = cb
        self._autoclose_figures = autoclose_figures

    def __enter__(self):
        self._cb(0)
        return self

    def __call__(self, value):
        return self._cb(value)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._cb(1)
        if self._autoclose_figures:
            plt.close('all')


def get_cb(cb, suffix=None):
    # Define the callback as a no-op if not provided or sets up tqdm if requested.
    if cb is None:
        cb = lambda x: x
    elif cb == 'tqdm':
        from tqdm import tqdm
        mesg = '{l_bar}{bar}[{elapsed}<{remaining}]'
        if suffix is not None:
            mesg = mesg + ' ' + suffix
        pbar = tqdm(total=100, bar_format=mesg)
        def cb(frac):
            nonlocal pbar
            frac *= 100
            pbar.update(frac - pbar.n)
            if frac == 100:
                pbar.close()
    else:
        raise ValueError(f'Unsupported callback: {cb}')
    return cb


def add_default_options(parser):
    parser.add_argument('folder', type=str, help='Folder containing data')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess all data in folder')
    parser.add_argument('--halt-on-error', action='store_true', help='Stop on error?')


def process_files(folder, glob_pattern, fn, cb='tqdm', reprocess=False,
                  halt_on_error=False):
    success = []
    errors = []
    for filename in Path(folder).glob(glob_pattern):
        if filename.suffix == '.md5':
            continue
        try:
            fn(filename, cb=cb, reprocess=reprocess)
            success.append(filename)
        except KeyboardInterrupt:
            # Don't capture this otherwise it just keeps continuing with the
            # next file.
            raise
        except Exception as e:
            if halt_on_error:
                raise
            errors.append((filename, e))
            print(f'Error processing {filename}')
    print(f'Successfully processed {len(success)} files with {len(errors)} errors')


def add_trial(df, grouping):
    def _add_trial(df):
        df['trial'] = range(len(df))
        return df.set_index('trial', append=True)
    result = df.groupby(grouping, group_keys=False).apply(_add_trial)
    return result


def cal_from_epl(name, base_path=None):
    if base_path is None:
        base_path = Path('c:/Data/Probe Tube Calibrations')
    filename = base_path / f'{name}_ProbeTube.calib'
    with filename.open('r') as fh:
        for line in fh:
            if line.startswith('Freq(Hz)'):
                break
        cal = pd.read_csv(fh, sep='\t',
                          names=['freq', 'SPL', 'phase'])
    return InterpCalibration.from_spl(cal['freq'], cal['SPL'],
                                      phase=cal['phase'])


class DatasetManager:

    def __init__(self, path, raw_dir=None, proc_dir=None, file_template=None):
        '''
        Manages paths of processed files given the relative path between the
        raw and processed directory structure.

        Parameters
        ----------
        raw_dir : {None, str, Path}
            Base path containing raw data
        proc_dir : {None, str, Path}
            Base path containing processed data
        file_template : {None, str}
            If None, defaults to the filename stem
        '''
        if raw_dir is None:
            raw_dir = os.environ.get('RAW_DATA_DIR', None)
        if proc_dir is None:
            proc_dir = os.environ.get('PROC_DATA_DIR', None)
        self.path = Path(path)
        self.raw_dir = Path(raw_dir)
        self.proc_dir = Path(proc_dir)
        if file_template is None:
            file_template = f'{self.path.stem}'
        self.file_template = file_template

    def create_cb(self, cb):
        return CallbackManager(get_cb(cb, self.path.stem))

    def get_proc_path(self):
        return self.proc_dir / self.path.parent.relative_to(self.raw_dir) / self.path.stem

    def get_proc_filename(self, suffix, mkdir=True):
        proc_path = self.get_proc_path()
        proc_path.mkdir(exist_ok=True, parents=True)
        return proc_path / f'{self.file_template} {suffix}'

    def is_processed(self, suffixes):
        if isinstance(suffixes, str):
            suffixes = [suffixes]
        for suffix in suffixes:
            if not self.get_proc_filename(suffix).exists():
                print(f'... {suffix} missing')
                return False
        return True

    def save_dict(self, d, suffix):
        filename = self.get_proc_filename(suffix)
        filename.write_text(json.dumps(d))

    def save_fig(self, figure, suffix):
        filename = self.get_proc_filename(suffix)
        figure.savefig(filename, bbox_inches='tight')

    def save_figs(self, figures, suffix):
        filename = self.get_proc_filename(suffix).with_suffix('.pdf')
        with PdfPages(filename) as pdf:
            for figure in figures:
                pdf.savefig(figure, bbox_inches='tight')

    def save_dataframe(self, df, suffix, **kw):
        filename = self.get_proc_filename(suffix)
        df.to_csv(filename, **kw)

    save_df = save_dataframe

    def clear(self, suffixes):
        for suffix in suffixes:
            filename = self.get_proc_filename(suffix)
            if filename.exists():
                filename.unlink()
