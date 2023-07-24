from psidata.api import Recording


class EFR(Recording):

    def __init__(self, filename, setting_table='analyze_efr_metadata'):
        super().__init__(filename, setting_table)
        self.efr_type = 'ram' if 'efr_ram' in self.base_path.stem else 'sam'

    def _get_epochs(self, signal):
        duration = self.get_setting('duration')
        offset = 0
        result = signal.get_epochs(self.analyze_efr_metadata, offset, duration)
        if self.efr_type == 'sam':
            # Drop the requested fc and fm column. The actual fc and fm are
            # stored in target_sam_tone_fc and target_sam_tone_fm (e.g., we may
            # coerce the fc period to an integer divisor of the stimulus
            # duration).
            result = result.reset_index(['fc', 'fm'], drop=True)
            rename = {
                'target_sam_tone_fc': 'fc',
                'target_sam_tone_fm': 'fm',
            }
            result.index.names = [rename.get(n, n) for n in result.index.names]
            return result
        else:
            # Note that duty cycle is also a computed parameter, so
            # double-check this if needed.
            result = result.reset_index(['fc', 'fm'], drop=True)
            rename = {
                'target_tone_frequency': 'fc',
                'target_mod_fm': 'fm',
            }
            result.index.names = [rename.get(n, n) for n in result.index.names]
            return result

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
