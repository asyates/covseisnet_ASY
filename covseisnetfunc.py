#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import covseisnet as csn
import obspy
import matplotlib.pyplot as plt
from usefulFuncs import *
from obspy import UTCDateTime
from obspy import Stream
from obspy import read
import numpy as np
import os
from datetime import datetime


def run_covseisnet(folder, channel, startdate, enddate, writeoutdir, average=50, window_duration_sec =100, dfac = 5, ss=False, ss_stat='', printstream=False):

    currentdate = UTCDateTime(startdate)
    enddate = UTCDateTime(enddate)

    numdays = int((enddate - currentdate)/86400)+1 #+1 to include enddate
   
    times_all = np.zeros(numdays, dtype=object)
    sw_all = np.zeros(numdays, dtype=object)

    for i in range(numdays):


        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Processing day %d, year %d, from data in folder %s' % (currentdate.julday, currentdate.year, folder))  

        st = getDayWaveform(folder, channel, currentdate, ss, ss_stat)
       
        if len(st) == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': WARNING - No traces pulled into stream. Going to next day') 
            currentdate = currentdate + 86400
            continue
        else:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': %d traces pulled into stream. Pre-processing data...' % (len(st))) 

        if printstream==True:
            print(st)

        #pre-processing
        preProcessStream(st, currentdate, dfac)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Pre-processing finished, computing spectral width.')

        #compute spectral width
        try:
            times, frequencies, spectral_width = computeSpectralWidth(st, window_duration_sec, average) 

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Finished computing spectral width, saving result') 
            saveCovOutput(writeoutdir, currentdate, times, frequencies, spectral_width, len(st))
        except Exception as e:
            print('Error computing covariance matrix / spectral width')
            print(str(e))

        #increment date
        currentdate = currentdate + 86400       
  

def plotSpectralWidth(directory, startdate, enddate, samprate = 20, log=True, count=False):

    currentdate = UTCDateTime(startdate)
    enddate = UTCDateTime(enddate)

    numdays = int((enddate - currentdate)/86400)+1 #+1 to include enddate

    times_all = np.zeros(numdays, dtype=object)
    sw_all = np.zeros(numdays, dtype=object)
    statcount_all = np.zeros(numdays,dtype=object)
    for i in range(numdays):
       
        try: 
            times, frequencies, spectral_width, statcount = readCovOutput(directory, currentdate, count)
        except:
            times = np.zeros(times.shape)
            spectral_width = np.zeros(spectral_width.shape)
        
        times_all[i] = times +(86400*i) #multiply by i+1 so that times are increasing    
        sw_all[i] = spectral_width.T
        statcount_all[i] = statcount
    
        #if appending new set of times, remove last time of previous entry (as corresponds to first of new)
        if i>0:             
            times_all[i-1] = times_all[i-1][:-1] 
        
        #increment date
        currentdate = currentdate + 86400 

    #stack arrays horizontally
    times = np.hstack(times_all)
    spectral_width = np.hstack(sw_all)

    if count == True:
        #show network covariance matrix spectral width
        fig, ax = plt.subplots(2, constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})
        plot_sw(fig, ax[0], times, frequencies, spectral_width, samprate, log)
        days = np.arange(numdays)
        ax[1].bar(days, statcount_all, width=1.0, align='edge')
        ax[1].set_xlim(0,numdays) 
    else:
        fig, ax = plt.subplots(1, constrained_layout=True)
        plot_sw(fig, ax, times, frequencies, spectral_width, samprate, log)

    plt.show()

def plot_sw(fig, ax, times, frequencies, spectral_width, samprate, log):

    img = ax.pcolormesh(times/86400, frequencies, spectral_width, rasterized=True, cmap="viridis_r", shading='auto')
    ax.set_ylim([0.01, samprate/2])
    
    if log==True:
        ax.set_yscale('log')

    fig.colorbar(img, ax=ax)
    fig.colorbar(img, ax=ax).set_label("Covariance matrix spectral width")

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Hz)")


def preProcessStream(st, currentdate, dfac):

    #downsample data to 20 Hz 
    st.decimate(dfac)
    maxpts = len(max(st,key=len))

    startdatetime = UTCDateTime(currentdate.year, currentdate.month, currentdate.day)   
   
    #synchronise
    st.synchronize(start=startdatetime, duration_sec = 86400, method="linear")

    #remove stations with missing data
    for tr in st:
        if len(tr) < maxpts - 1:
            st.remove(tr)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Trace with missing data removed, %d traces remaining' % len(st))

    #preprocess using smooth spectral whitening and temporal normalization
    st.preprocess(domain="spectral", method="smooth")
    st.preprocess(domain="temporal", method="smooth")

    

def computeSpectralWidth(st, window_duration_sec, average):

    #calculate covariance from stream
    times, frequencies, covariances = csn.covariancematrix.calculate(st, window_duration_sec, average)

    #calculate spectral width
    spectral_width = covariances.coherence(kind="spectral_width")

    return times, frequencies, spectral_width

def getDayWaveform(folder, channel, date, ss, ss_stat):

    datapath = 'data/'+folder
   
    print(datapath)

    st_jday = date.julday
    st_year = date.year
    
    st = csn.arraystream.ArrayStream()

    if channel == '*':
        channel = ''

    for root, dirs, files in os.walk(datapath+'/'+str(st_year)):
        for file in files:
             
            if file.endswith(channel+'.D.'+str(st_year)+'.'+str(st_jday)+'.MSEED'):
                if ss==False:
                    st += read(root+'/'+file)
                else:
                    if file.find(ss_stat) != -1:
                        st += read(root+'/'+file)
    st.merge(fill_value=0)
    
    return st   

def readCovOutput(directory, date, statcount):
        
    filename = str(date.year)+'_'+str(date.julday)+'.npy'
    
    covresult = np.load(directory+'/'+filename, allow_pickle=True)
 
    times = covresult[0]
    frequencies = covresult[1]
    spectral_width = covresult[2]
    if statcount==True:
        statcount = covresult[3]
    else:
        statcount = np.nan

    return times, frequencies, spectral_width, statcount

def saveCovOutput(directory, date, times, freq, spectral_width, statcount):
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    output = np.zeros(4, dtype=object)

    output[0] = times
    output[1] = freq
    output[2] = spectral_width
    output[3] = statcount
    filename = str(date.year)+'_'+str(date.julday)
    
    np.save(directory+'/'+filename, output)


#OLD FUNCTION (NOT CURRENTLY USED)
def getWaveformsFromDir(folder, channel, startdate,enddate):

    datapath = '/home/yatesal/Data/Seismic/'+folder
    
    st_jday, st_year = convertDateToYearJulian(startdate)
    en_jday, en_year = convertDateToYearJulian(enddate)

    cu_year = st_year
    cu_jday = st_jday

    st = csn.arraystream.ArrayStream()

    while (cu_year != en_year or cu_jday != en_jday):

        st_temp = Stream()

        for root, dirs, files in os.walk(datapath+'/'+str(cu_year)):
            for file in files:
                if file.endswith(channel+'.D.'+str(cu_year)+'.'+str(cu_jday)):
                    st_temp += read(root+'/'+file)
        st_temp.merge(fill_value=0)
        st += st_temp

        cu_jday += 1
        if cu_jday > 366:
            cu_year += 1
            cu_jday = 1
    
    return st

