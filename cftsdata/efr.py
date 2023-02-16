from psidata.api import Recording


class EFR(Recording):

    def __init__(self, filename, setting_table='analyze_efr_metadata'):
        super().__init__(filename, setting_table)

    def _get_epochs(self, signal):
        duration = self.get_setting('duration')
        offset = 0
        result = signal.get_epochs(self.analyze_efr_metadata, offset, duration)
        return result.reset_index(['target_sam_tone_fc', 'target_sam_tone_fm'],
                                  drop=True)

    @property
    def mic(self):
        return self.system_microphone

    def get_eeg_epochs(self):
        return self._get_epochs(self.eeg)

    def get_mic_epochs(self):
        return self._get_epochs(self.mic)
