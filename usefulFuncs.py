#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def convertDateStrToYearJulian(date):
    
    fmt = '%Y-%m-%d'
    dt = datetime.datetime.strptime(date, fmt)

    #convert to time tuple for converting to julian day
    tt = dt.timetuple()
    jday = tt.tm_yday
    year = tt.tm_year

    return jday, year

def convertDateStrToDatetime(date):

    fmt = '%Y-%m-%d'
    dt = datetime.datetime.strptime(date,fmt)

    return dt

def convertDatetime64ToStr(date):

    ts = pd.to_datetime(str(date))
    d = ts.strftime('%Y-%m-%d')
    return d

def fillArray(value, length): 
    arr = np.zeros(length, dtype=object)
    for i in range(length):
        arr[i] = value
    return arr

def smoothdata(data, N):
    print('smoothing data') 
   
    halfwin = int((N-1)/2) #make sure N is odd

    smoothed_data = np.zeros(len(data))
    for i in range(len(data)):

        if i < N/2:
            smoothed_data[i] = np.nan
        else:
            smoothed_data[i] = np.nanmean(data[i-halfwin:i+halfwin+1])
   
    return smoothed_data

def getPercentZero(data):
    
    non_zeros = np.count_nonzero(data)
    percentzero = 1 - (float(non_zeros)/len(data))

    return percentzero
    


def smooth(x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
            
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
         in the begining and end part of the output signal.
                                
        """
        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")
        if window_len<3:
            return(x)

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'bl              ackman'")

        #print window_len                                         
        s=np.r_[x[int(window_len)-1:0:-1],x,x[-2:-int(window_len)-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')

        #return to original size of x
        yfix = y[(int(window_len/2)-1):-(int(window_len/2))]

        if len(yfix) > len(x):
            yfix = yfix[1:]

        return yfix

def getDistance(lat1, lon1, lat2, lon2):
    #return distance between two points in km

    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance
