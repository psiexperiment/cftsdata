import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2

from psiaudio import util
from psiaudio import plot
from psiaudio import stats

from . import memr
from .util import add_default_options, process_files


th_expected_suffixes = [
    'HT2 threshold diagnostics.pdf',
    'HT2 2samp.csv',
    'HT2 threshold.json',
    'HT2 processing settings.json',
    'HT2 raw amplitude.csv',
    'HT2 fitted amplitude.csv',
]


# Functions derived from Suthakar and Liberman 2019
sigmoid = lambda x, a, b, c, d: a + b / (1 + np.exp(np.exp(d) * (c-x)))
power = lambda x, a, b, c: a * x ** b + c


def get_fit_text(a, b, c, d):
    return fr'${a:.2f} + \frac{{{b:.2f}}}{{1 + e^{{e^{{{d:.2f}}} \cdot ({c:.2f} - x)}}}}$'


def sigmoid_norm(p, x, y, loss):
    y_pred = sigmoid(x, *p)
    z = (y - y_pred) ** 2
    if loss == 'linear':
        loss = z
    elif loss == 'soft_l1':
        loss = 2 * ((1 + z)**0.5 - 1)
    elif loss == 'cauchy':
        loss = np.log(1 + z)
    elif loss == 'arctan':
        loss = np.arctan(z)
    return np.sum(loss)


def fit_sigmoid(x, y, th_criterion, asymptote_criterion=None, loss='soft_l1'):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    # Offset, amplitude, threshold, slope
    guess = [min(y), (max(y)-min(y)), np.mean(x), -0.5]
    # Clip the last parameter to 0 to limit the slope of the sigmoid to
    # reasonable values for the elicitor level scale.
    bounds = [(0, np.inf), (0, np.inf), (min(x), max(x)), (-10, 0)]
    result = optimize.minimize(sigmoid_norm, guess, bounds=bounds, args=(x, y, loss))
    x_pred = np.linspace(min(x), max(x), 1000)
    fit = pd.Series(sigmoid(x_pred, *result.x), index=x_pred)
    fit.index.name = 'elicitor_level'

    th = np.interp(th_criterion, fit.values, fit.index)
    result = {
        'p': result.x,
        'fit': fit,
        'th_criterion': th_criterion,
        'threshold': th,
        'formula': get_fit_text(*result.x),

    }

    if asymptote_criterion is not None:
        result.update({
            'asymptote': np.interp(asymptote_criterion, fit.index, fit.values),
            'asymptote_criterion': asymptote_criterion,
        })
    return result


def _process_th(manager, probe_norm, t2_2samp, proc_info):
    # Generate the color mapping that we will use for plotting
    elicitor_levels = probe_norm.index.unique('elicitor_level')
    color_map = {e: c for c, e in plot.iter_colors(elicitor_levels)}
    figure, axes = plt.subplots(2, 3, constrained_layout=True, figsize=(10.5, 8))

    memr_db = util.db(np.abs(probe_norm))
    memr_db_total = util.db(np.abs(probe_norm - 1) + 1)

    # Compute the average MEMR amplitude across the full frequency range
    # (using the total MEMR).
    average_memr = util.db(util.dbi(memr_db_total).mean(axis=1))
    max_memr = util.db(util.dbi(memr_db_total).max(axis=1))

    # Plot conventional MEMR
    for l, m in memr_db.iterrows():
        axes[0, 0].plot(m, color=color_map[l], label=str(l))
    axes[0, 0].set_xscale('octave')
    axes[0, 0].set_xlabel('Frequency (kHz)')
    axes[0, 0].set_ylabel('MEMR amplitude (dB SPL)')
    axes[0, 0].axis(ymin=-3, ymax=3)
    axes[0, 0].set_title('Classic MEMR')

    # Plot total MEMR
    for l, m in memr_db_total.iterrows():
        axes[0, 1].plot(m, color=color_map[l])
    axes[0, 1].set_xscale('octave')
    axes[0, 1].set_xlabel('Frequency (kHz)')
    axes[0, 1].set_ylabel('MEMR amplitude (dB SPL)')
    axes[0, 1].axis(ymin=0, ymax=3)
    axes[0, 1].set_title('Total MEMR')

    # Plot MEMR groth functions
    axes[0, 2].plot(average_memr.index, average_memr.values, 'k-', label='Average')
    axes[0, 2].plot(max_memr.index, max_memr.values, '-', color='seagreen', label='Max')
    axes[0, 2].set_xlabel('Elicitor Level')
    axes[0, 2].set_ylabel('MEMR amplitude (dB SPL)')
    axes[0, 2].legend()
    axes[0, 2].axis(ymin=0, ymax=3)
    axes[0, 2].set_title('Total MEMR amplitude')

    axes[1, 0].plot(average_memr, 'ko')
    axes[1, 0].set_title('Average MEMR amplitude')
    axes[1, 0].set_ylabel('MEMR amplitude (dB SPL)')

    average_memr_fit = fit_sigmoid(average_memr.index, average_memr, 0.25, 80)
    axes[1, 0].text(0.05, 0.95, average_memr_fit['formula'],
                    transform=axes[1, 0].transAxes, ha='left', va='top')
    axes[1, 0].plot(average_memr_fit['fit'], 'r-')
    axes[1, 0].axvline(average_memr_fit['threshold'], color='r', ls=':')
    axes[1, 0].axhline(average_memr_fit['asymptote'], color='r', ls=':')

    axes[1, 1].plot(max_memr, 'ko')
    axes[1, 1].set_title('Max MEMR amplitude')
    axes[1, 1].set_ylabel('MEMR amplitude (dB SPL)')
    max_memr_fit = fit_sigmoid(max_memr.index, max_memr, 0.5, 80)
    axes[1, 1].text(0.05, 0.95, max_memr_fit['formula'],
                    transform=axes[1, 1].transAxes, ha='left', va='top')
    axes[1, 1].plot(max_memr_fit['fit'], 'r-')
    axes[1, 1].axvline(max_memr_fit['threshold'], color='r', ls=':')
    axes[1, 1].axhline(max_memr_fit['asymptote'], color='r', ls=':')

    memr_amplitude = pd.DataFrame({
        'max': max_memr,
        'mean': average_memr,
    })

    fitted_memr_amplitude = pd.DataFrame({
        'max': max_memr_fit.get('fit', None),
        'mean': average_memr_fit.get('fit', None),
    })

    # Calculate the equivalent F-value for the criterion p-value
    axes[1, 2].plot(t2_2samp['F'], 'ko')
    axes[1, 2].set_title('2-sample $F$ statistic')
    axes[1, 2].set_ylabel('$F$ statistic')
    df = t2_2samp['df'].iloc[0]
    t2_2samp_p_crit = chi2.isf(0.5, df=df)
    t2_2samp_fit = fit_sigmoid(t2_2samp.index, t2_2samp['F'], t2_2samp_p_crit)
    axes[1, 2].text(0.05, 0.95, t2_2samp_fit['formula'],
                    transform=axes[1, 2].transAxes, ha='left', va='top')
    axes[1, 2].plot(t2_2samp_fit['fit'], 'r-')
    axes[1, 2].axvline(t2_2samp_fit['threshold'], color='r', ls=':')
    axes[1, 2].axhline(t2_2samp_p_crit, color='r', ls=':')

    for ax in axes[-1]:
        ax.set_xlabel('Elicitor Level (dB SPL)')

    manager.save_df(memr_amplitude, 'HT2 raw amplitude.csv')
    manager.save_df(fitted_memr_amplitude, 'HT2 fitted amplitude.csv')
    manager.save_fig(figure, 'HT2 threshold diagnostics.pdf')
    manager.save_df(t2_2samp, 'HT2 2samp.csv')
    manager.save_dict({
        'mean_threshold': average_memr_fit['threshold'],
        'max_threshold': max_memr_fit['threshold'],
        'mean_asymptote': average_memr_fit['asymptote'],
        'max_asymptote': max_memr_fit['asymptote'],
        't2_2samp_threshold': t2_2samp_fit['threshold'],
        't2_2samp_threshold_crit': t2_2samp_p_crit,
    }, 'HT2 threshold.json')
    manager.save_dict({
        **proc_info,
    }, 'HT2 processing settings.json')


def compute_ht2_2samp_keefe(x, train):
    x = np.abs(x)
    train = np.abs(train)
    test = x.loc[1:]
    result = stats.ht2_2samp(train, test)
    return pd.Series(result, index=result._fields)


def process_keefe_th(filename, manager, freq_lb=5.6e3, freq_ub=16e3,
                     turntable_speed=1.25, min_corr=0.9):
    kwargs = dict(locals())
    kwargs.pop('filename')
    kwargs.pop('manager')

    with manager.create_cb():
        fh = memr.InterleavedMEMRFile(filename)

        # Load only the valid probes
        probe_valid = fh.valid_epochs(fh.get_probe(),
                                      turntable_speed=turntable_speed,
                                      min_corr=min_corr)

        # Compute the Hotelling T^2 statistic from the CSD
        probe_csd = util.csd_df(probe_valid, fs=fh.probe_fs).loc[:, freq_lb:freq_ub]
        t2_2samp = probe_csd.groupby('elicitor_level') \
            .apply(compute_ht2_2samp_keefe, train=probe_csd.loc[0])

        # Compute the MEMR for plotting
        probe_norm = (probe_csd / probe_csd.loc[0]).groupby('elicitor_level').mean()

        _process_th(manager, probe_norm, t2_2samp, kwargs)


def compute_ht2_2samp_valero(x, train=None):
    x = np.abs(x)
    if train is None:
        try:
            train = x.xs('baseline', level='group')
        except KeyError:
            return pd.Series({'T2': np.nan, 'F': np.nan, 'p': np.nan, 'df': np.nan})
    else:
        train = np.abs(train)
    test = x.xs('elicitor', level='group')
    result = stats.ht2_2samp(train, test)
    return pd.Series(result, index=result._fields)


def process_valero_th(filename, manager, freq_lb=5.6e3, freq_ub=16e3,
                      turntable_speed=1.25, min_corr=0.9, max_ht2=160):
    kwargs = dict(locals())
    kwargs.pop('filename')
    kwargs.pop('manager')

    with manager.create_cb():
        fh = memr.SimultaneousMEMRFile(filename)

        # Load only the valid probes
        probe_valid = fh.valid_epochs(fh.get_probe(trim=(0, 1e-3)),
                                      turntable_speed=turntable_speed,
                                      min_corr=min_corr,
                                      max_ht2=max_ht2)

        # Compute the Hotelling T^2 statistic from the CSD
        probe_csd = util.csd_df(probe_valid, fs=fh.probe_fs).loc[:, freq_lb:freq_ub]
        t2_2samp = probe_csd.groupby('elicitor_level').apply(compute_ht2_2samp_valero)

        # Compute the MEMR for plotting
        probe_csd_mean = probe_csd.groupby(['group', 'elicitor_polarity', 'elicitor_level']).mean()
        probe_norm = probe_csd_mean.loc['elicitor'] / probe_csd_mean.loc['baseline']
        probe_norm = probe_norm.groupby('elicitor_level').mean()

        _process_th(manager, probe_norm, t2_2samp, kwargs)


def main_keefe_th():
    import argparse
    parser = argparse.ArgumentParser('Estimate threshold for Valero MEMR data')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_interleaved*',
                  fn=process_keefe_th,
                  expected_suffixes=th_expected_suffixes, **args)


def main_valero_th():
    import argparse
    parser = argparse.ArgumentParser('Estimate threshold for Valero MEMR data')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_simultaneous*',
                  fn=process_valero_th,
                  expected_suffixes=th_expected_suffixes, **args)
