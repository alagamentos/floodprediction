import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def derivative_threshold(timeSeries, threshold, plot = True, plt_start = None, plt_stop = None, ylim = None):
    """ 
    inputs:
        timeSeries - Pandas Series
        threshold - Maxmium derivative absolute value
        plot - Bool - plots timeSeries and its derivative
        plt_start - start index
        plt_plt_stop - plt_stop index

    Returns:
        peaks - boolean list - True for over threshold and False for under
    """ 
    # Derivative
    timeSeries = timeSeries.fillna(0)
    d_ts = np.gradient(timeSeries)

    # Find peaks
    peaks = []
    for i in range(len(d_ts)):
        peaks.append(abs(d_ts[i]) > threshold)


    # Plot data
    if plot:
        abs_ = [abs(i) for i in (d_ts[plt_start:plt_stop])]
        lim = roundup(max(abs_))
        plt.figure(figsize =  (12,7))
        ax1 = plt.subplot(211)
        ax1.plot(timeSeries.index[plt_start:plt_stop],timeSeries[plt_start:plt_stop])
        if ylim:
            ax1.set_ylim(ylim[0],ylim[1])
        i = 0
        for d in peaks:
            if d and i > plt_start and i < plt_stop:
                ax1.axvline(i, ymin=-1.1*lim, ymax=1.1*lim, c = 'r')
            i = i + 1
        ax2 = plt.subplot(212)
        ax2.axhline(threshold , c = 'r')
        ax2.axhline(-threshold, c = 'r') 
        ax2.plot(timeSeries.index[plt_start:plt_stop],d_ts[plt_start:plt_stop])
        plt.show()

    return peaks

def derivative_zero(timeSeries, n_zeros, plot = False, plt_start = None, plt_stop = None, ylim = None):
    """ 
    For a index i, search in a region [i:i+n_zeros] and [i-n_zeros:i] if derivative is zero.
    const[i] is True if all values inside regions are zeros.

    Inputs:
        timeSeries - Pandas Series
        n_zeros - Number of consecutive zeros necessary
        plot - Bool - plots timeSeries and its derivative
        plt_start - start index
        plt_plt_stop - plt_stop index

    Output:
        const - boolean list - True for when multiple consecutive zeros
    """ 
    # Derivative
    timeSeries = timeSeries.fillna(0)
    d_ts = np.gradient(timeSeries)

    # Find consecutive zeros
    const = []
    for i in range(len(d_ts)):
        aux = True
        for n in range(n_zeros):
            if (i+n >= len(d_ts)):
                aux = aux and d_ts[i - n] == 0
            elif (i-n < 0):
                aux = aux and d_ts[i + n] == 0
            else:
                aux = aux and (d_ts[i + n] == 0 or d_ts[i - n] == 0)
        const.append(aux)

    # Plot data
    if plot:
        lim = roundup(max(abs(d_ts[plt_start:plt_stop])))
        if lim == 0:
            lim = 10
        plt.figure(figsize =  (12,7))
        ax1 = plt.subplot(211)
        i = 0
        for d in const:
            if d and i > plt_start and i < plt_stop:
                ax1.axvline(i, ymin=-lim, ymax=lim , c = 'g', alpha = 0.25)
            i = i + 1
        ax1.plot(timeSeries.index[plt_start:plt_stop],timeSeries[plt_start:plt_stop])
        if ylim:
            ax1.set_ylim(ylim[0],ylim[1])
        ax2 = plt.subplot(212)
        ax2.plot(timeSeries.index[plt_start:plt_stop],d_ts[plt_start:plt_stop])
        ax2.set_yticks(np.arange(-lim ,lim + 1,10))
        ax2.set_ylim(-lim ,lim )
        ax1.set_title('Time series')
        ax2.set_title('Derivative')
        plt.show()

    return const


def list_2_regions(bool_list):
    """
    Input:
        bool_list - List of booleans

    Output:
        regions - [[start, end], [start, end]] - List of lists with the starting and ending
        index of consecutives True values

    """
    regions = []
    i = 0
    status = False
    for bool_ in bool_list:
        if bool_ and not status:
            start = i
            status = True
        if not bool_ and status:
            end = i
            status = False
            regions.append([start,end])
        i += 1

    return regions

def regions_2_list(regions, total_len):
    """

    """
    bool_list = [False] * total_len
    for reg in regions:
        for i in range(reg[0],reg[1]):
            bool_list[i] = True

    return bool_list


def increase_margins(margin, regions, total_len):
    """

    """
    regions_marg = []
    for reg in regions:
        start = reg[0] - margin
        stop = reg[1] + margin
        if stop > total_len:
            stop = total_len
        if start < 0:
            start = 0
        regions_marg.append([start, stop])

    return regions_marg

def plot_regions(timeSeries, regions, total_len, start, stop, plt_type = 'region', title = None, ylim = None):
    """

    """
    plt.figure(figsize =  (12,4))
    plt.plot(timeSeries.index[start:stop],timeSeries[start:stop])
    plt.title(title)

    for reg in regions:
        if (reg[0] >= start and reg[0] <= stop) or (reg[1] >= start and reg[1] <= stop) or \
        (reg[0] <= start and reg[1] >= stop):
            if reg[0] <= start and (reg[1] >= start and reg[1] <= stop):
                plt_start = start
                plt_stop = reg[1]
            elif (reg[0] >= start and reg[0] <= stop) and reg[1] >= stop :
                plt_start = reg[0]
                plt_stop = stop
            elif (reg[0] <= start and reg[1] >= stop):
                plt_start = start
                plt_stop = stop
            else:
                plt_start = reg[0]
                plt_stop = reg[1]
            if plt_type == 'region':    
                plt.axvspan(plt_start , plt_stop, color = 'red')
            else:
                x = list(range(plt_start,plt_stop))
                y = timeSeries[plt_start:plt_stop]
                plt.plot(x, y, color = 'red')
                if ylim:
                    plt.ylim(ylim[0],ylim[1])

    plt.show()
