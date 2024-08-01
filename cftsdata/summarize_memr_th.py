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


keefe_th_expected_suffixes = [
    'HT2 threshold diagnostics.pdf',
    'HT2 individual.csv',
    'HT2 2samp.csv',
    'HT2 threshold.json',
    'raw amplitude.csv',
    'fitted amplitude.csv',
]


# Functions derived from Suthakar and Liberman 2019
sigmoid = lambda x, a, b, c, d: a + ((b-a) / (1 + np.exp(d * (c-x))))
power = lambda x, a, b, c: a * x ** b + c


def fit_sigmoid(series, th_criterion, asymptote_criterion=None):
    x = series.index.values
    y = series.values
    def fit_fn(p):
        nonlocal x
        nonlocal y
        return y - sigmoid(x, *p)
    guess = [min(y), max(y), np.mean(x), 1]
    bounds = (
        [0, 0, min(x), -np.inf],
        [np.inf, np.inf, max(x), np.inf],
    )
    result = optimize.least_squares(fit_fn, guess, bounds=bounds)
    x_pred = np.linspace(min(x), max(x), 1000)
    fit = pd.Series(sigmoid(x_pred, *result.x), index=x_pred)
    fit.index.name = series.index.name

    th = np.interp(th_criterion, fit.values, fit.index)
    if not result.success:
        raise ValueError('Optimization failed')
    result = {
        'p': result.x,
        'fit': fit,
        'th_criterion': th_criterion,
        'threshold': th,
    }

    if asymptote_criterion is not None:
        result.update({
            'asymptote': np.interp(asymptote_criterion, fit.index, fit.values),
            'asymptote_criterion': asymptote_criterion,
        })
    return result


def compute_ht2_individual(x, train=None, train_with_elicitor=False):
    x = np.abs(x)
    if train is None:
        # Default to the elicitor-off condition for only this elicitor level
        train = x.loc[0]
    else:
        # Use the provided training data.
        train = np.abs(train)

    test = x.loc[1:]

    if train_with_elicitor:
        train = np.concatenate((train, test))

    result = np.abs(stats.ht2_individual(train, test))
    return pd.Series(result, index=test.index)


def compute_ht2_2samp(x, train):
    x = np.abs(x)
    train = np.abs(train)
    test = x.loc[1:]
    result = stats.ht2_2samp(train, test)
    return pd.Series(result, index=result._fields)


def plot_t2(t2_individual, t2_2samp, ax, color_map):
    for l, t in t2_individual.groupby('elicitor_level'):
        jitter = np.random.uniform(-1.5, 1.5, size=len(t))
        ax.plot(l + jitter, t, 'o', color=color_map[l], mec='w', mew=1)

    t2_mean = t2_individual.groupby('elicitor_level').mean()
    t2_min = t2_individual.groupby('elicitor_level').min()
    ax.plot(t2_min, '-', label='Minimum $T^2$ statistic\nfor each level', color='blueviolet')
    ax.plot(t2_mean, '-', label='Mean $T^2$ statistic\nfor each level', color='cornflowerblue')
    if t2_2samp is not None:
        ax.plot(t2_2samp, '-', label='2-sample $T^2$ statistic\nfor each level', color='darkcyan')
    ax.set_yscale('log')
    ax.set_xlabel('Elicitor level (dB SPL)')
    ax.set_ylabel('$T^2$ statistic')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


def process_keefe_th(filename, manager, freq_lb=5.6e3, freq_ub=16e3,
                     t2_criterion=1000):
    with manager.create_cb():
        fh = memr.InterleavedMEMRFile(filename)

        # Load only the valid probes
        probe_valid = fh.valid_epochs(fh.get_probe())

        # Generate the color mapping that we will use for plotting
        elicitor_levels = probe_valid.index.unique('elicitor_level')
        color_map = {e: c for c, e in plot.iter_colors(elicitor_levels)}
        figure, axes = plt.subplots(2, 4, constrained_layout=True, figsize=(18, 8))

        # Compute the Hoetelling T^2 statistic from the CSD
        probe_csd = util.csd_df(probe_valid, fs=fh.probe_fs).loc[:, freq_lb:freq_ub]
        t2_2samp = probe_csd.groupby('elicitor_level') \
            .apply(compute_ht2_2samp, train=probe_csd.loc[0])
        t2_individual = probe_csd.loc[1:].groupby('elicitor_level') \
            .apply(compute_ht2_individual, train=probe_csd.loc[0])

        # Calculate threshold using the minimum
        t2_min = t2_individual.groupby('elicitor_level').min()
        t2_mean = t2_individual.groupby('elicitor_level').mean()

        # Compute the MEMR for plotting
        probe_norm = (probe_csd / probe_csd.loc[0]).groupby('elicitor_level').mean()
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
        axes[0, 2].set_ylabel('MEMR amplitude')
        axes[0, 2].legend()
        axes[0, 2].axis(ymin=0, ymax=3)
        axes[0, 2].set_title('Total MEMR amplitude')

        plot_t2(t2_individual, t2_2samp['T2'], axes[0, 3], color_map)
        axes[0, 3].set_title('Test for deviation from elicitor-off')

        average_memr_fit = fit_sigmoid(average_memr, 0.25, 80)
        axes[1, 0].plot(average_memr, 'ko')
        axes[1, 0].plot(average_memr_fit['fit'], 'r-')
        axes[1, 0].axvline(average_memr_fit['threshold'], color='r', ls=':')
        axes[1, 0].axhline(average_memr_fit['asymptote'], color='r', ls=':')
        axes[1, 0].set_xlabel('Elicitor level')
        axes[1, 0].set_title('Average MEMR amplitude')

        max_memr_fit = fit_sigmoid(max_memr, 0.5, 80)
        axes[1, 1].plot(max_memr, 'ko')
        axes[1, 1].plot(max_memr_fit['fit'], 'r-')
        axes[1, 1].axvline(max_memr_fit['threshold'], color='r', ls=':')
        axes[1, 1].axhline(max_memr_fit['asymptote'], color='r', ls=':')
        axes[1, 1].set_title('Max MEMR amplitude')

        memr_amplitude = pd.DataFrame({
            'max': max_memr,
            'mean': average_memr,
        })

        fitted_memr_amplitude = pd.DataFrame({
            'max': max_memr_fit['fit'],
            'mean': average_memr_fit['fit'],
        })

        t2_min_fit = fit_sigmoid(t2_min, t2_criterion)
        axes[1, 2].plot(t2_min, 'ko')
        axes[1, 2].plot(t2_min_fit['fit'], 'r-')
        axes[1, 2].axvline(t2_min_fit['threshold'], color='r', ls=':')
        axes[1, 2].set_title('Minimum $T^2$ statistic')

        # Calculate the equivalent F-value for the criterion p-value
        df = t2_2samp['df'].iloc[0]
        crit = chi2.isf(0.01, df=df)
        t2_2samp_fit = fit_sigmoid(t2_2samp['F'], crit)
        axes[1, 3].plot(t2_2samp['F'], 'ko')
        axes[1, 3].plot(t2_2samp_fit['fit'], 'r-')
        axes[1, 3].axvline(t2_2samp_fit['threshold'], color='r', ls=':')
        axes[1, 3].set_title('2-sample $F$ statistic')
        axes[1, 3].axhline(crit, color='r', ls=':')

        manager.save_df(memr_amplitude, 'raw amplitude.csv')
        manager.save_df(fitted_memr_amplitude, 'fitted amplitude.csv')
        manager.save_fig(figure, 'HT2 threshold diagnostics.pdf')
        manager.save_df(t2_individual.rename('statistic'), 'HT2 individual.csv')
        manager.save_df(t2_2samp, 'HT2 2samp.csv')
        manager.save_dict({
            'mean_threshold': average_memr_fit['threshold'],
            'max_threshold': max_memr_fit['threshold'],
            'mean_asymptote': average_memr_fit['asymptote'],
            'max_asymptote': max_memr_fit['asymptote'],
            't2_min_threshold': t2_min_fit['threshold'],
            't2_2samp_threshold': t2_2samp_fit['threshold'],
        }, 'HT2 threshold.json')


def main_keefe_th():
    import argparse
    parser = argparse.ArgumentParser('Estimate threshold for Valero MEMR data')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_pattern='**/*memr_interleaved*',
                  fn=process_keefe_th,
                  expected_suffixes=keefe_th_expected_suffixes, **args)
