'''
Validates ABR data on a new system

Connect speaker output to EEG input.
'''

import matplotlib.pyplot as plt
import numpy as np

from cftsdata import efr


def main(filename):
    fh = efr.EFR(filename)
    epochs = fh._get_epochs(fh.output_monitor)

    all_fm = epochs.index.unique('fm')
    all_fc = epochs.index.unique('fc')
    n_fm = len(all_fm)
    n_fc = len(all_fc)
    grouped = epochs.groupby(['fm', 'fc'])

    epochs_mean = epochs.groupby(['fm', 'fc']).mean()
    figure, axes = plt.subplots(n_fm, n_fc, figsize=(n_fc*4, n_fm*4))
    for row, fm in enumerate(all_fm):
        for col, fc in enumerate(all_fc):
            d = epochs_mean.loc[(fm, fc), :]
            axes[row, col].plot(d, color='black')
    figure.suptitle('All mean')

    figure, axes = plt.subplots(n_fm, n_fc, figsize=(n_fc*4, n_fm*4))
    for row, fm in enumerate(all_fm):
        for col, fc in enumerate(all_fc):
            d = grouped.get_group((fm, fc))
            a = d.xs(1, level='polarity').iloc[0]
            b = d.xs(1, level='polarity').iloc[-1]
            axes[row, col].plot(a, color='lightsalmon')
            axes[row, col].plot(b, color='seagreen')
            axes[row, col].plot(a-b, color='black')
    figure.suptitle('Last vs. first run')

    figure, axes = plt.subplots(n_fm, n_fc, figsize=(n_fc*4, n_fm*4))
    for row, fm in enumerate(all_fm):
        for col, fc in enumerate(all_fc):
            d = grouped.get_group((fm, fc))
            a = d.xs(1, level='polarity').iloc[0]
            b = d.xs(-1, level='polarity').iloc[-1]
            axes[row, col].plot(a, color='lightsalmon')
            axes[row, col].plot(b, color='seagreen')
            axes[row, col].plot(a+b, color='black')
    figure.suptitle('Pos vs. neg. pol.')

    plt.show()




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('validate-efr')
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args.filename)
