import numpy

from typing import Literal


def remove_outliers(times, outliers_mode: Literal['quantile', 'guassian', 'median'] = 'quantile') -> numpy.ndarray:
    outlier_flags = numpy.array([False for _ in range(len(times))])
    if outliers_mode == 'quantile':
        q1 = numpy.quantile(times, 0.25, axis=-1)
        q3 = numpy.quantile(times, 0.75, axis=-1)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        lower_outlier_flags = times < lower_whisker
        upper_outlier_flags = upper_whisker < times
        outlier_flags = lower_outlier_flags | upper_outlier_flags
    if outliers_mode == 'guassian':
        avg = numpy.average(times, axis=-1)
        std = numpy.std(times, axis=-1, ddof=1)
        outlier_flags = numpy.absolute(times - avg) > 3 * std
    if outliers_mode == 'median':
        med = numpy.median(times, axis=-1)
        mad = numpy.median(numpy.absolute(times - med), axis=-1)
        outlier_flags = numpy.absolute(times - med) > 3 * mad

    return times[~outlier_flags]


def get_outlier_flags(times, outliers_mode: Literal['quantile', 'guassian', 'median'] = 'quantile') -> numpy.ndarray:
    outlier_flags = numpy.array([False for _ in range(len(times))])
    if outliers_mode == 'quantile':
        q1 = numpy.quantile(times, 0.25, axis=-1)
        q3 = numpy.quantile(times, 0.75, axis=-1)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        lower_outlier_flags = times < lower_whisker
        upper_outlier_flags = upper_whisker < times
        outlier_flags = lower_outlier_flags | upper_outlier_flags
    if outliers_mode == 'guassian':
        avg = numpy.average(times, axis=-1)
        std = numpy.std(times, axis=-1, ddof=1)
        outlier_flags = numpy.absolute(times - avg) > 3 * std
    if outliers_mode == 'median':
        med = numpy.median(times, axis=-1)
        mad = numpy.median(numpy.absolute(times - med), axis=-1)
        outlier_flags = numpy.absolute(times - med) > 3 * mad

    return outlier_flags