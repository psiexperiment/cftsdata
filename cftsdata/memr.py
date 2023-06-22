from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from psidata.api import Recording
from . import util

# Max size of LRU cache
MAXSIZE = 1024


# Columns to potentially rename.
RENAME = {
    'probe_chirp_start_frequency': 'probe_fl',
    'probe_chirp_end_frequency': 'probe_fh',
    'probe_bandlimited_click_flb': 'probe_fl',
    'probe_bandlimited_click_fub': 'probe_fh',
    'probe_chirp_n': 'probe_n',
    'probe_click_n': 'probe_n',
    'probe_chirp_delay': 'probe_delay',
    'probe_click_delay': 'probe_delay',
    'probe_bandlimited_click_window': 'probe_duration',
    'elicitor_bandlimited_noise_fl': 'elicitor_fl',
    'elicitor_bandlimited_noise_fh': 'elicitor_fh',
    'elicitor_bandlimited_noise_polarity': 'elicitor_polarity',
    'elicitor_bandlimited_noise_level': 'elicitor_level',
}


class BaseMEMRFile(Recording):

    def __init__(self, base_path, setting_table='memr_metadata'):
        if 'memr' not in Path(base_path).stem:
            raise ValueError(f'{base_path} is not a MEMR recording')
        super().__init__(base_path, setting_table)

    @property
    def probe_fs(self):
        return self.probe_microphone.fs

    @property
    def elicitor_fs(self):
        return self.elicitor_microphone.fs

    @property
    def probe_microphone(self):
        # A refactor of the cfts suite resulted in microphone being renamed to
        # system_microphone.
        try:
            return self.__getattr__('probe_microphone')
        except AttributeError:
            return self.__getattr__('microphone')

    @property
    @lru_cache(maxsize=MAXSIZE)
    def memr_metadata(self):
        try:
            data = self.__getattr__('memr_metadata')
        except AttributeError:
            data = self.__getattr__('memr_probe_metadata')
        # We need to check what needs to be renamed since an update to the MEMR
        # paradigm now includes the renamed column names.
        rename = {k: v for k, v in RENAME.items() if v not in data}
        return data.rename(columns=rename)

    @lru_cache(maxsize=MAXSIZE)
    def get_epochs(self, columns='auto', signal_name='probe_microphone',
                   add_trial=True, cb=None):
        signal = getattr(self, signal_name)
        epochs = signal.get_epochs(
            self.memr_metadata, 0, self.trial_duration,
            columns=columns, cb=cb).sort_index()
        if add_trial:
            epochs = util.add_trial(epochs, epochs.index.names[:-1])
        return epochs

    @lru_cache(maxsize=MAXSIZE)
    def _get_repeats(self, columns='auto', signal_name='probe_microphone'):
        fs = getattr(self, signal_name).fs
        epochs = self.get_epochs(columns, signal_name).copy()
        s_repeat = int(round(self.repeat_period * fs))
        n_probe = self.get_setting('probe_n')
        t_probe = np.arange(s_repeat) / fs

        repeats = []
        keys = []
        for i in range(n_probe):
            lb = s_repeat * i
            ub = lb + s_repeat
            repeat = epochs.iloc[:, lb:ub]
            repeat.columns.values[:] = t_probe
            repeats.append(repeat)
            keys.append((i, lb / fs))
        return pd.concat(repeats, keys=keys, names=['repeat', 'probe_t0'])

    @property
    def trial_duration(self):
        raise NotImplementedError

    @property
    def repeat_period(self):
        raise NotImplementedError


class InterleavedMEMRFile(BaseMEMRFile):

    @property
    def trial_duration(self):
        return self.get_setting('probe_n') * self.get_setting('repeat_period')

    @property
    def repeat_period(self):
        return self.get_setting('repeat_period')

    @lru_cache(maxsize=MAXSIZE)
    def get_elicitor(self, signal_name='elicitor_microphone'):
        repeats = self._get_repeats(signal_name=signal_name)
        elicitor_delay = self.get_setting('elicitor_envelope_start_time')
        m = repeats.columns >= elicitor_delay
        return repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)

    @lru_cache(maxsize=MAXSIZE)
    def get_probe(self, acoustic_delay=0.75e-3, signal_name='probe_microphone'):
        repeats = self._get_repeats(signal_name=signal_name)
        probe_delay = self.get_setting('probe_delay')
        probe_duration = self.get_setting('probe_duration')
        probe_lb = acoustic_delay + probe_delay
        probe_ub = acoustic_delay + probe_delay + probe_duration
        m = (repeats.columns >= probe_lb) & (repeats.columns < probe_ub)
        return repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)

    @lru_cache(maxsize=MAXSIZE)
    def get_silence(self, acoustic_delay=0.75e-3, signal_name='probe_microphone'):
        probe_delay = self.get_setting('probe_delay')
        probe_duration = self.get_setting('probe_duration')
        silence_lb = acoustic_delay + probe_delay + probe_duration
        silence_ub = silence_lb + probe_duration
        repeats = self._get_repeats(signal_name=signal_name)
        m = (repeats.columns >= silence_lb) & (repeats.columns < silence_ub)
        return repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)


class SimultaneousMEMRFile(BaseMEMRFile):

    @property
    def trial_duration(self):
        return self.get_setting('trial_duration')

    @property
    def repeat_period(self):
        return 1 / self.get_setting('probe_rate')

    def get_repeats(self, columns='auto', signal_name='probe_microphone',
                    norm_window=None):
        repeats = super().get_repeats(columns, signal_name)

        probe_n = self.get_setting('probe_n')
        onset = self.get_setting('elicitor_onset')
        duration = self.get_setting('elicitor_duration')
        rise = self.get_setting('elicitor_noise_rise_time')
        if norm_window is None:
            norm_window = self.get_setting('norm_window')

        def to_repeat(x):
            return int(round(x / self.repeat_period))

        # Mark elicitor portions
        e_start = to_repeat(onset + rise)
        e_end = to_repeat(onset + duration - rise)

        # Norm window is just before the elicitor begins
        nw_start = to_repeat(onset - norm_window)
        nw_end = to_repeat(onset)

        # Create a mapping of repeat number to the probe type (e.g., baseline,
        # elicitor, recovery).
        probe_map = pd.Series('', index=range(probe_n))
        probe_map[e_start:e_end] = 'elicitor'
        probe_map[nw_start:nw_end] = 'baseline'
        probe_map[e_end:] = 'recovery'

        ix = repeats.index.to_frame(index=False)
        ix['epoch_t0'] = ix['t0']
        ix['t0'] = ix.eval('epoch_t0 + probe_t0')
        ix['group'] = ix['repeat'].map(probe_map)
        new_names = repeats.index.names[:-1] + ['group']

        names = list(repeats.index.names)
        names.remove('repeat')
        names.remove('t0')
        names.remove('probe_t0')
        names = names + ['t0', 'epoch_t0', 'probe_t0', 'repeat', 'group']

        repeats.index = ix.set_index(names).index
        return repeats.sort_index()
