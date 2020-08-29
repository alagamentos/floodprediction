import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

C1 = '#1a9988'
C2 = '#eb5600'


def roundup(x):
  return int(math.ceil(x / 10.0)) * 10


def derivative_threshold(timeSeries, threshold, plot=True,
                         plt_start=None, plt_stop=None, ylim=None,
                         figsize=(12, 7), lw=2):
  """
  Returns a boolean list with true for every sample with the derivative
  greater then the defined threshold

  Inputs:
    timeSeries - Pandas Series
    threshold - Maxmium derivative absolute value
    plot - Bool - plots timeSeries and its derivative
    plt_start - start index
    plt_plt_stop - plt_stop index
    figsize - matplotlib figure size
    lw = matplotlib linewidth

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
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(212)
    ax1.plot(timeSeries.index[plt_start:plt_stop],
             timeSeries[plt_start:plt_stop], c=C1, lw=lw)
    if ylim:
      ax1.set_ylim(ylim[0], ylim[1])

    i = 0

    for d in peaks:
      if d and i > plt_start and i < plt_stop:
        ax1.axvline(i, ymin=-1.1*lim, ymax=1.1*lim, c=C2, lw=lw)

      i = i + 1

    ax2 = plt.subplot(211)
    ax2.axhline(threshold, c=C2, lw=lw)
    ax2.axhline(-threshold, c=C2, lw=lw)
    ax2.plot(timeSeries.index[plt_start:plt_stop],
             d_ts[plt_start:plt_stop], c=C1, lw=lw)
    ax1.set_title('Time series')
    ax2.set_title('Derivative')
    plt.show()

  return peaks


def derivative_zero(timeSeries, n_zeros, non_zero=False,
                    plot=False, plt_start=None, plt_stop=None,
                    ylim=None, figsize=(12, 7), lw=2):
  """
  For a index i, search in a region [i:i+n_zeros] and [i-n_zeros:i] if derivative is zero.
  const[i] is True if all values inside regions are zeros.

  Inputs:
    timeSeries - Pandas Series
    n_zeros - Number of consecutive zeros necessary
    non_zero -
    plot - Bool - plots timeSeries and its derivative
    plt_start - start index
    plt_plt_stop - plt_stop index
    figsize - matplotlib figure size
    lw = matplotlib linewidth

  Returns:
    const - boolean list - True for when multiple consecutive zeros
  """

  # Derivative
  timeSeries = timeSeries.fillna(0)
  timeSeries = timeSeries.reset_index(drop=True)
  d_ts = np.gradient(timeSeries)

  # Find consecutive zeros
  const = []

  for i in range(len(d_ts)):
    start = i-(n_zeros//2)
    stop = i+(n_zeros//2)
    if start < 0:
        start = 0
    if stop > len(timeSeries):
        stop = len(timeSeries)
    const.append((d_ts[start:stop] == 0).all())


  if non_zero:
    not_zeros = []
    not_zeros = timeSeries != 0

    const = [not_zeros[i] and const[i] for i in range(len(not_zeros))]

  # Plot data
  if plot:
    lim = roundup(max(abs(d_ts[plt_start:plt_stop])))

    if lim == 0:
      lim = 10
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(212)
    i = 0

    for d in const:
      if d and i > plt_start and i < plt_stop:
        ax1.axvline(i, ymin=-lim, ymax=lim, c=C2, alpha=1)

      i = i + 1

    ax1.plot(timeSeries.index[plt_start:plt_stop],
             timeSeries[plt_start:plt_stop], lw=lw, c=C1)

    if ylim:
      ax1.set_ylim(ylim[0], ylim[1])

    ax2 = plt.subplot(211)
    ax2.plot(timeSeries.index[plt_start:plt_stop],
             d_ts[plt_start:plt_stop], lw=lw, c=C1)
    ax2.set_yticks(np.arange(-lim, lim + 1, 10))
    ax2.set_ylim(-lim, lim)
    ax2.axhline(0, c=C2, lw=lw)
    ax1.set_title('Time series')
    ax2.set_title('Derivative')
    plt.show()

  return const


def list_2_regions(bool_list):
  """
  Input:
    bool_list - List of booleans

  Returns:
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
    elif not bool_ and status:
      end = i
      status = False
      regions.append([start, end])

    i += 1

  return regions


def regions_2_list(regions, total_len):
  """
  Inputs:
    regions - List of lists - each internal list contains
    the start and end index for a given boolean list

    total_len - Integer - length of list/dataframe

  Returns:
    bool_list - list of booleans respective to given regions
  """

  bool_list = [False] * total_len

  for reg in regions:
    for i in range(reg[0], reg[1]):
      bool_list[i] = True

  return bool_list


def increase_margins(margin, regions, total_len):
  """
  Increase the margins of a given regions (list of lists)

  Inputs:
    margin - How much samples to increase (each side - start and stop)

    regions - List of lists - each internal list contains
    the start and end index for a given boolean list

    total_len - Integer - length of list/dataframe

  Returns:

    regions_marg - List of lists - same as regions inputs but with
    increased regions given by margin.
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


def plot_regions(timeSeries, regions, start, stop,
                 plt_type='region', title=None, ylim=None, figsize=(12, 7), lw=2):
  """
  Plot a given regions input with blue fore correct data and red for false.
  This function does not return any value, but plots as matplot.pyplot figure

  Inputs:
    timeSeries - plot data
    regions - List of lists - each internal list contains
          the start and end index for a given boolean list
    start, stop - Ingeter - Start and stop ploting index
    plt_type - 'region': changes the background color for the incorrect data
          other  : changes the plot color for the incorrect data
    title - matplotlib plot title
    ylim - matplotlib y plot loimits
    figsize - matplotlib figure size
    lw = matplotlib linewidth
  """

  total_len = len(timeSeries)

  plt.figure(figsize=figsize)
  plt.plot(timeSeries.index[start:stop], timeSeries[start:stop], lw=lw, c=C1)
  plt.title(title)

  for reg in regions:
    if (reg[0] >= start and reg[0] <= stop) or (reg[1] >= start and reg[1] <= stop) or \
            (reg[0] <= start and reg[1] >= stop):
      if reg[0] <= start and (reg[1] >= start and reg[1] <= stop):
        plt_start = start
        plt_stop = reg[1]
      elif (reg[0] >= start and reg[0] <= stop) and reg[1] >= stop:
        plt_start = reg[0]
        plt_stop = stop
      elif (reg[0] <= start and reg[1] >= stop):
        plt_start = start
        plt_stop = stop
      else:
        plt_start = reg[0]
        plt_stop = reg[1]

      if plt_type == 'region':
        plt.axvspan(plt_start, plt_stop, color=C2)
      else:
        x = list(range(plt_start, plt_stop))
        y = timeSeries[plt_start:plt_stop]
        plt.plot(x, y, color='red', lw=lw)

        if ylim:
          plt.ylim(ylim[0], ylim[1])

  plt.show()
