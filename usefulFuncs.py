#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np

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
