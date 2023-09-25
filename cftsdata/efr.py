from functools import lru_cache

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

    def _get_epochs(self, signal, columns='auto'):
        duration = self.get_setting('duration')
        return signal.get_epochs(self.analyze_efr_metadata, 0, duration, columns=columns)

    @property
    def mic(self):
        return self.system_microphone

    def get_eeg_epochs(self, columns='auto'):
        return self._get_epochs(self.eeg, columns=columns)

    def get_mic_epochs(self, columns='auto'):
        return self._get_epochs(self.mic, columns=columns)

    @property
    def level(self):
        if self.efr_type == 'ram':
            return self.get_setting('target_tone_level')
        else:
            return self.get_setting('target_sam_tone_level')
