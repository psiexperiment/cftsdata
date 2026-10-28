from functools import lru_cache

import pandas as pd

from psidata.api import Recording


MAXSIZE = 1024


class EFR(Recording):

    def __init__(self, filename, setting_table='analyze_efr_metadata'):
        super().__init__(filename, setting_table)
        self.efr_type = 'ram' if 'efr_ram' in self.base_path.stem else 'sam'

    @property
    @lru_cache(maxsize=MAXSIZE)
    def analyze_efr_metadata(self):
        '''
        EFR metadata in DataFrame format

        There will be one row for each epoch and one column for each parameter
        from the EFR experiment. For simplicity, some parameters have been
        renamed so that we have `fc`, `fm` and `polarity`.
        '''
        data = self.__getattr__('analyze_efr_metadata')
        drop = [c for c in ('fc', 'fm') if c in data]
        result = data.drop(columns=drop) \
            .rename(columns={
            'target_sam_tone_fc': 'fc',
            'target_sam_tone_fm': 'fm',
            'target_sam_tone_polarity': 'polarity',
            'target_tone_frequency': 'fc',
            'target_mod_fm': 'fm',
            'target_tone_polarity': 'polarity',
        })

        # In some of the earliest experiments, polarity was not included in the
        # analyze_efr_metadata file.
        if 'polarity' not in result:
            result['polarity'] = 1

        return result

    def _get_epochs(self, signal, downsample=None, segment_duration=None, columns='auto'):
        duration = self.get_setting('duration')
        result = signal.get_epochs(self.analyze_efr_metadata, 0, duration,
                                   downsample=downsample, columns=columns)
        if segment_duration is not None:
            n_segments = self.get_setting('duration') / segment_duration
            if n_segments != int(n_segments):
                raise ValueError('Cannot segment into {n_segments} equally-sized segments')
            n_segments = int(n_segments)

            reshaped = {}
            for key, row in result.iterrows():
                time = row.index.values
                signal = row.values
                time = time.reshape((n_segments, -1))
                signal = signal.reshape((n_segments, -1))
                signal = pd.DataFrame(signal, columns=time[0], index=time[:, 0])
                signal.index.name = 't0_segment'
                signal.columns.name = 'time'
                reshaped[key] = signal
            reshaped = pd.concat(reshaped, names=result.index.names)
            reshaped.attrs.update(result.attrs)
            return reshaped

        return result

    @property
    def mic(self):
        return self.system_microphone

    def get_eeg_epochs(self, target_fs=None, segment_duration=None,
                       columns='auto'):
        if target_fs is not None:
            n_dec = int(self.eeg.fs // target_fs)
        else:
            n_dec = None
        return self._get_epochs(self.eeg, downsample=n_dec,
                                segment_duration=segment_duration,
                                columns=columns)

    def get_mic_epochs(self, segment_duration=None, columns='auto'):
        return self._get_epochs(self.mic, segment_duration=segment_duration, columns=columns)

    @property
    def level(self):
        if self.efr_type == 'ram':
            return self.get_setting('target_tone_level')
        else:
            return self.get_setting('target_sam_tone_level')
