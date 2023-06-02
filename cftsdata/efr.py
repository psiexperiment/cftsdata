from psidata.api import Recording


class EFR(Recording):

    def __init__(self, filename, setting_table='analyze_efr_metadata'):
        super().__init__(filename, setting_table)
        self.efr_type = 'ram' if 'efr_ram' in self.base_path.stem else 'sam'

    def _get_epochs(self, signal):
        duration = self.get_setting('duration')
        offset = 0
        result = signal.get_epochs(self.analyze_efr_metadata, offset, duration)
        print(result)
        if self.efr_type == 'sam':
            to_drop = ['target_sam_tone_fc', 'target_sam_tone_fm']
            return result.reset_index(to_drop, drop=True)
        else:
            to_drop = ['target_mod_fm', 'target_tone_frequency', 'target_mod_duty_cycle']
        return result.reset_index(to_drop, drop=True)

    @property
    def mic(self):
        return self.system_microphone

    def get_eeg_epochs(self):
        return self._get_epochs(self.eeg)

    def get_mic_epochs(self):
        return self._get_epochs(self.mic)

    @property
    def level(self):
        if self.efr_type == 'ram':
            return self.get_setting('target_tone_level')
        else:
            return self.get_setting('target_sam_tone_level')
