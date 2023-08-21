from psiaudio import util
from psidata.api import Recording


class InearCalibration(Recording):

    def __init__(self, base_path, setting_table='epoch_metadata'):
        super().__init__(base_path, setting_table)

    def get_epochs(self):
        duration = self.get_setting('hw_ao_chirp_duration')
        return self.hw_ai.get_epochs(self.epoch_metadata, 0, duration)

    def get_average_psd(self, apply_calibration=True):
        epochs = self.get_epochs()
        epochs_mean = epochs.groupby(epochs.index.names[:-1]).mean()
        psd = util.psd_df(epochs_mean, self.hw_ai.fs)
        if apply_calibration:
            cal = self.hw_ai.get_calibration()
            return cal.get_db(psd)
        return util.db(psd)
