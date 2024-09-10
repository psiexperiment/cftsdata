#from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from psiaudio import stats
from psiaudio.util import psd_df
from psidata.api import Recording
from . import util


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

    def get_velocity(self, signal_name='turntable_linear_velocity'):
        raise NotImplementedError

    def get_speed(self, signal_name='turntable_linear_velocity'):
        return np.abs(self.get_velocity(signal_name))

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
    def memr_metadata(self):
        try:
            data = self.__getattr__('memr_metadata')
        except AttributeError:
            data = self.__getattr__('memr_probe_metadata')
        # We need to check what needs to be renamed since an update to the MEMR
        # paradigm now includes the renamed column names.
        rename = {k: v for k, v in RENAME.items() if v not in data}
        return data.rename(columns=rename)

    def get_epochs(self, columns='auto', signal_name='probe_microphone',
                   add_trial=True, cb=None):
        signal = getattr(self, signal_name)
        epochs = signal.get_epochs(
            self.memr_metadata, 0, self.trial_duration,
            columns=columns, cb=cb).reset_index('elicitor_uuid', drop=True) \
            .sort_index()
        if add_trial:
            dtype = epochs.columns.dtype
            # This is converting a float column index to an object index.
            epochs = util.add_trial(epochs, epochs.index.names[:-1])
            epochs.columns = epochs.columns.astype(dtype)
        return epochs

    def get_repeats(self, columns='auto', signal_name='probe_microphone'):
        fs = getattr(self, signal_name).fs
        epochs = self.get_epochs(columns, signal_name).copy()

        t_repeat = self.repeat_period * fs
        s_repeat = max(1, int(round(t_repeat)))

        n_probe = self.get_setting('probe_n')
        t_probe = np.arange(s_repeat) / fs

        repeats = []
        keys = []
        for i in range(n_probe):
            lb = int(round(t_repeat * i))
            ub = lb + s_repeat
            repeat = epochs.iloc[:, lb:ub]
            repeat.columns.values[:] = t_probe
            repeats.append(repeat)
            keys.append((i, lb / fs))
        return pd.concat(repeats, keys=keys, names=['repeat', 'probe_t0'])

    def get_probe(self, acoustic_delay=0.75e-3, signal_name='probe_microphone',
                  trim=0):
        if isinstance(trim, tuple):
            trim_lb, trim_ub = trim
        else:
            trim_lb = trim_ub = trim
        probe_delay = self.get_setting('probe_delay')
        probe_duration = self.get_setting('probe_duration')
        probe_lb = acoustic_delay + probe_delay + trim_lb
        probe_ub = acoustic_delay + probe_delay + probe_duration + trim_ub
        if probe_ub <= probe_lb:
            raise ValueError('Bad values for trim')

        repeats = self.get_repeats(signal_name=signal_name)
        m = (repeats.columns >= probe_lb) & (repeats.columns < probe_ub)

        drop = [d for d in ('probe_t0', 't0') if d in repeats.index.names]
        return repeats.loc[:, m].reset_index(drop, drop=True)

    @property
    def trial_duration(self):
        raise NotImplementedError

    @property
    def repeat_period(self):
        raise NotImplementedError


class InterleavedMEMRFile(BaseMEMRFile):

    def get_velocity(self, signal_name='turntable_linear_velocity'):
        return self.get_epochs(signal_name=signal_name)

    @property
    def trial_duration(self):
        return self.get_setting('probe_n') * self.get_setting('repeat_period')

    @property
    def repeat_period(self):
        return self.get_setting('repeat_period')

    def get_elicitor(self, signal_name='elicitor_microphone'):
        repeats = self.get_repeats(signal_name=signal_name)
        elicitor_delay = self.get_setting('elicitor_envelope_start_time')
        m = repeats.columns >= elicitor_delay
        return repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)

    def get_silence(self, acoustic_delay=0.75e-3,
                    signal_name='probe_microphone', trim=0):
        probe_delay = self.get_setting('probe_delay')
        probe_duration = self.get_setting('probe_duration')
        silence_lb = acoustic_delay + probe_delay + probe_duration + trim
        silence_ub = silence_lb + probe_duration - trim
        if silence_ub <= silence_lb:
            raise ValueError('Trim value set too high')

        repeats = self.get_repeats(signal_name=signal_name)
        m = (repeats.columns >= silence_lb) & (repeats.columns < silence_ub)
        return repeats.loc[:, m].reset_index(['probe_t0', 't0'], drop=True)

    def get_max_epoch_speed(self):
        # Load the turntable speed and find maximum across entire epoch. We
        # drop the very last sample because these samples have not always been
        # available in the online artifact reject so we want to be sure we
        # don't end up rejecting a trial that was kept in the online artifact
        # reject.
        return self.get_speed().iloc[:, :-1] \
            .max(axis=1).reset_index('t0', drop=True)

    def get_valid_epoch_mask(self, turntable_speed=None):
        # If turntable_speed is not set, load from the settings.
        if turntable_speed is None:
            turntable_speed = self.get_setting('max_turntable_speed')
        valid = self.get_max_epoch_speed() < turntable_speed
        valid.name = 'valid'
        return valid

    def valid_epochs(self, epochs, trial_n=None, turntable_speed=None,
                     min_corr=None):

        speed_valid = self.get_valid_epoch_mask(turntable_speed)

        if min_corr is None:
            min_corr = self.get_setting('min_probe_corr')
        corr = epochs.groupby(['elicitor_polarity', 'elicitor_level', 'trial']) \
            .apply(lambda x: np.corrcoef(x.values).min())
        corr_valid = corr >= min_corr

        valid = corr_valid & speed_valid

        if trial_n is None:
            trial_n = int(self.get_setting('trial_n') / 2)
        if (trial_n * 2) != self.get_setting('trial_n'):
            raise ValueError('Unequal number of positive and negative polarity trials')

        # This artifact reject is designed to reject a full trial, not just
        # individual probes. We also want to keep only the desired number of
        # trials and not include more (for consistency in analysis).
        grouping = ['elicitor_level', 'elicitor_polarity']
        result = epochs.unstack('repeat').loc[valid] \
            .groupby(grouping, group_keys=False) \
            .apply(lambda x: x.iloc[:trial_n]) \
            .stack('repeat')
        result.index = result.index.reorder_levels((3, 0, 1, 2))
        return result.sort_index()


class SimultaneousMEMRFile(BaseMEMRFile):

    def get_max_epoch_speed(self):
        # Load the turntable speed and find maximum across entire epoch. For
        # the default settings in cfts, the stimulus is already
        return self.get_speed().max(axis=1)

    def get_valid_epoch_mask(self, epochs=None, turntable_speed=None,
                             min_corr=None, max_ht2=np.inf):
        '''
        Parameters
        ----------
        max_ht2 : float
            Implement outlier detection based on Hotelling's T^2 using the
            specified value as a cutoff. To disable this check, set `max_ht2`
            to `np.inf`.
        '''
        if epochs is None:
            epochs = self.get_probe()

        # If turntable_speed is not set, load from the settings.
        if turntable_speed is None:
            turntable_speed = self.get_setting('max_turntable_speed')
        speed = self.get_max_epoch_speed().loc[epochs.index]
        valid = speed < turntable_speed
        valid.name = 'valid'

        # Get average correlation of epoch with all other epochs.
        if min_corr is None:
            min_corr = self.get_setting('min_probe_corr')
        cc = np.corrcoef(epochs)
        cc_valid = np.corrcoef(epochs).mean(axis=0) > min_corr
        valid &= cc_valid

        epochs_psd = psd_df(epochs, fs=self.probe_fs).loc[:, 4e3:32e3]
        ht2_valid = stats.ht2_individual(epochs_psd) < max_ht2
        valid &= ht2_valid

        return valid

    def valid_epochs(self, epochs, turntable_speed=None, min_corr=None,
                     max_ht2=np.inf):
        '''
        Parameters
        ----------
        max_ht2 : float
            Implement outlier detection based on Hotelling's T^2 using the
            specified value as a cutoff. To disable this check, set `max_ht2`
            to `np.inf`.
        '''
        valid = self.get_valid_epoch_mask(epochs, turntable_speed, min_corr,
                                          max_ht2)
        return epochs.loc[valid]

    @property
    def trial_duration(self):
        return self.get_setting('trial_duration')

    def get_velocity(self, signal_name='turntable_linear_velocity'):
        repeats = self.get_repeats(signal_name=signal_name)
        return repeats.reset_index(['probe_t0', 't0'], drop=True)

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


class SweepMEMRFile(BaseMEMRFile):

    def get_epochs(self, *args, **kwargs):
        epochs = super().get_epochs(*args, **kwargs, add_trial=False).sort_index(level='t0')
        epochs['trial'] = np.arange(len(epochs))
        return epochs.set_index(['trial'], append=True)

    def get_repeats(self, signal_name='probe_microphone'):
        r = super().get_repeats(signal_name=signal_name) \
            .reset_index(['elicitor_polarity', 'probe_t0', 't0'], drop=True)
        r.index = r.index.swaplevel('repeat', 'trial')
        return r.sort_index()

    @property
    def trial_duration(self):
        return self.get_setting('trial_duration')

    @property
    def repeat_period(self):
        return 1 / self.get_setting('probe_rate')

    @property
    def repeat_t0(self):
        pass

    @property
    def repeat_n(self):
        return int(self.trial_duration * self.get_setting('probe_rate'))

    def repeat_t(self, extra_delay=0.75e-3):
        offset = fh.get_setting('probe_click_delay') + extra_delay + fh.get_setting('probe_duration') * 0.5
        return np.arange(self.repeat_n) * self.repeat_period + offset
