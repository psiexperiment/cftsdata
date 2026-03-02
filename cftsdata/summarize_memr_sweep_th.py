from scipy.optimize import curve_fit
from scipy.stats import norm

from cftsdata.memr import SweepMEMRFile


sweep_expected_suffixes = [
    'MEMR_threshold.pdf',
    'MEMR_amplitude.csv',
    'MEMR_threshold_stats.json'
]


def get_memr_db_total(filename):
    fh = SweepMEMRFile(filename)
    probe = fh.get_probe()
    settings = get_sweep_settings(fh)

    r_csd = util.csd_df(probe, fs=fh.probe_fs, window='hann').loc[:, 4e3:32e3]
    r_csd_dt = r_csd.apply(sweep_detrend, fs=fh.probe_fs)
    _, _, memr_db_total = csd_to_swept_memr(r_csd_dt)

    probe_times = np.arange(settings['probe_n'])/settings['probe_rate']
    probe_times -= settings['trial_duration']/2
    lb, ub = settings['elicitor_min_level'], settings['elicitor_max_level']
    elicitor_level = (ub-lb) - np.abs(probe_times) * settings['ramp_rate'] + settings['elicitor_min_level']
    elicitor_level = pd.Series(elicitor_level, name='elicitor_level')
    elicitor_level.index.name = 'repeat'
    memr_db_total = memr_db_total.join(elicitor_level).set_index('elicitor_level', append=True)
    memr_db_total.columns.name = 'frequency'

    return memr_db_total


def gaussian(x, mu, sigma, *a, ravel=True):
    a = np.array(a)[np.newaxis]
    pdf = norm.pdf(x, mu, sigma)
    z = a * pdf[:, np.newaxis] / pdf.max()
    if ravel:
        return z.ravel()
    return z


def fit_gaussian(data):
    x = np.arange(len(data))
    y = data.values
    p0 = [80, 24] + ([100] * y.shape[-1])
    popt, pcov = curve_fit(gaussian, x, y.ravel(), p0)
    y_pred = gaussian(x, *popt, ravel=False)
    y_pred = pd.DataFrame(y_pred, index=data.index, columns=data.columns)
    center_click, width, *amplitude = popt
    amplitude = pd.Series(amplitude, index=data.columns)
    return center_click, width, amplitude, y_pred


def plot_fit(memr_db_total, memr_db_pred, amplitude, threshold):
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    axes[0].axvline(threshold, ls=':', color='seagreen')

    axes[0].plot(memr_db_pred.iloc[:, ::10].values, 'k:')
    axes[0].plot(memr_db_total.loc[:, :16000].iloc[:, ::10].values, 'k-')
    axes[1].plot(amplitude, 'k-')
    axes[1].set_xscale('octave', octaves=0.5)

    axes[0].set_xlabel('Click Number')
    axes[1].set_xlabel('Frequency (kHz)')
    axes[0].set_ylabel('Amplitude (dB re baseline)')
    return figure


def process_sweep_file(filename, manager, freq_ub=16000, **kwargs):
    with manager.create_cb() as cb:
        memr_db_total = get_memr_db_total(filename)
        center_click, width, amplitude, memr_db_pred = fit_gaussian(memr_db_total.loc[:, :freq_ub])
        m = (memr_db_pred > 0.25).any(axis=1)
        _, threshold = memr_db_pred.loc[m].iloc[0].name
        figure = plot_fit(memr_db_total, memr_db_pred, amplitude, threshold)
        stats = {
            'amplitude_max': amplitude.max(),
            'amplitude_mean': amplitude.loc[8e3:freq_ub].mean(),
            'center': center_click,
            'width': width,
            'threshold': threshold,
        }

        manager.save_fig(figure, 'MEMR_threshold.pdf')
        manager.save_df(ampitude, 'MEMR_threshold_amplitude.csv')
        manager.save_mapping(stats, 'MEMR_threshold_stats.json')


def main_sweep():
    import argparse
    parser = argparse.ArgumentParser('Calculate MEMR sweep threshold')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_sweep*',
                  fn=process_sweep_file,
                  expected_suffixes=sweep_expected_suffixes, **args)

