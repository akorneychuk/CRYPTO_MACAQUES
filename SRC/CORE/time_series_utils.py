#!/usr/bin/env python
# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
from datetime import datetime
from xmlrpc.client import DateTime

import numpy as np


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


async def aggregate_thresholding_algo(data, queue_peaks, close_time: datetime, lag, threshold, influence):
    if len(data) >= lag:

        result = thresholding_algo(data, lag=lag, threshold=threshold, influence=influence)

        last_signal = result['signals'][-1]
        last_avgFilter = result['avgFilter'][-1]
        last_stdFilter = result['stdFilter'][-1]
        upper_stdFilter = last_avgFilter + threshold * last_stdFilter
        lower_stdFilter = last_avgFilter - threshold * last_stdFilter

        peak = {
            'close_time': close_time,
            'signal': int(last_signal),
            'avg': last_avgFilter,
            'std': last_stdFilter,
            'upper_std': upper_stdFilter,
            'lower_std': lower_stdFilter,
        }

        await queue_peaks.put(peak)