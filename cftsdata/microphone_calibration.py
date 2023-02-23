from psiaudio import util
from psidata.api import Recording


class MicrophoneCalibration(Recording):

    def get_psd(self, window='flattop', average_seconds=None):
        if average_seconds is not None:
            waveform_averages = int(self.hw_ai.duration // average_seconds)
        else:
            waveform_averages = None
        return util.db(util.psd_df(self.hw_ai[0], self.hw_ai.fs,
                                   waveform_averages=waveform_averages,
                                   window=window))
