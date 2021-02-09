#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import covseisnet as csn
import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from usefulFuncs import *
from obspy import UTCDateTime
from obspy import Stream
from obspy import read
import numpy as np
import os
from datetime import datetime, timedelta

def run_covseisnet(folder, channel, startdate, enddate, writeoutdir, average=100, window_duration_sec =100, dfac = 4, norm='onebit', spectral = 'onebit', stations=[], printstream=False):

    currentdate = UTCDateTime(startdate)
    enddate = UTCDateTime(enddate)

    numdays = int((enddate - currentdate)/86400)+1 #+1 to include enddate
   
    times_all = np.zeros(numdays, dtype=object)
    sw_all = np.zeros(numdays, dtype=object)

    for i in range(numdays):

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Processing day %d, year %d, from data in folder %s' % (currentdate.julday, currentdate.year, folder))  

        try:
            st = getDayWaveform(folder, channel, currentdate, stations)
        except Exception as e:
            print('Error reading data')
            print(str(e))
            currentdate = currentdate + 86400
            continue

        if len(st) == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': WARNING - No traces pulled into stream. Going to next day') 
            currentdate = currentdate + 86400
            continue
        else:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': %d traces pulled into stream. Pre-processing data...' % (len(st))) 

        if printstream==True:
            print(st)


        #for each hour of data
        tmpwinsize = 21600
        numtmpwin = int(86400/tmpwinsize)
        statcountarray = np.zeros(numtmpwin)
        timesarray = np.empty(numtmpwin, dtype=object)
        swarray = np.empty(numtmpwin, dtype=object)
        tmpcount = 0
        lstindex = -1

        st.trim(starttime=currentdate,endtime=currentdate+86400,fill_value=0, pad=True)  

        for tmp_st in st.slide(window_length=tmpwinsize, step=tmpwinsize):
            
            #pre-processing
            tmp_st = preProcessStream(tmp_st, currentdate+(tmpcount*tmpwinsize), dfac, norm, spectral, freqmin=0.01, freqmax=10, st_size = tmpwinsize)
            
            if len(tmp_st) == 0 or len(tmp_st) == 1:
                print('Error: Zero or one stream left after pre-processing, skipping day')
                #currentdate = currentdate + 86400
                statcountarray[tmpcount] = len(tmp_st)
                tmpcount = tmpcount+1
                continue
            else:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Pre-processing finished, computing spectral width.')
                if printstream==True:
                    print(tmp_st)

            #compute spectral width
            try:
                times, frequencies, spectral_width = computeSpectralWidth(tmp_st, window_duration_sec, average) 
                
                timesarray[tmpcount] = times + (tmpcount*tmpwinsize)
                swarray[tmpcount] = spectral_width.T
                statcountarray[tmpcount] = len(tmp_st)

                if lstindex != -1:
                    timesarray[lstindex] = timesarray[lstindex][:-1]
                lstindex = tmpcount
                
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Finished computing spectral width') 
                
            except Exception as e:
                print('Error computing covariance matrix / spectral width')
                print(str(e))

            tmpcount = tmpcount+1
        
        timesarray = np.array([x for x in timesarray if x is not None])
        swarray = np.array([x for x in swarray if x is not None])

        if len(timesarray) == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': No data for this day, skipping...')
            currentdate = currentdate + 86400  
            continue;
        
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Saving spectral width result for day')
        times = np.hstack(timesarray)
        spectral_width = np.hstack(swarray)
        saveCovOutput('outputs/'+writeoutdir, currentdate, times, frequencies, spectral_width, statcountarray)

        #increment date
        currentdate = currentdate + 86400    

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': ***FINISHED***')


def preProcessStream(st, currentdate, dfac, norm, spectral, freqmin=0.01, freqmax=10, st_size=86400):
 
    #remove stations with missing data
    maxpts = len(max(st,key=len))
    for tr in st:   
        print(getPercentZero(tr.data)) 
        if len(tr) < (maxpts * 0.99) or getPercentZero(tr.data) > 0.01: #allow 1% missing data or less than 5% zeroed data 
            st.remove(tr)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': Trace with missing data removed, %d traces remaining' % len(st))

    if len(st) != 0:
                
        #downsample data
        st.decimate(dfac)

        #synchronise
        st = st.synchronize(start=currentdate, duration_sec = st_size, method="linear")

        #preprocess using smooth spectral whitening and temporal normalization
        st.taper(0.05)
        st.detrend('linear')
        st.detrend('demean')
        st.preprocess(domain="spectral", method=spectral)
        st.preprocess(domain="temporal", method=norm)
        st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)
    
    return st

def computeSpectralWidth(st, window_duration_sec, average):

    #calculate covariance from stream
    times, frequencies, covariances = csn.covariancematrix.calculate(st, window_duration_sec, average)

    #calculate spectral width
    spectral_width = covariances.coherence(kind="spectral_width")

    return times, frequencies, spectral_width


def plotSpectralWidth(directory, startdate, enddate, winlenhr=6, log=True, count=False, norm=False, fig=None, ax=None):

    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)+1
    startdate = UTCDateTime(startdate)
    enddate = UTCDateTime(enddate)

    times, frequencies, spectral_width, statcount_all = getSpectralWidthData(directory, startdate, enddate, count)

    #if no ax or fig has been passed as argument, set to show plot at end of this function
    if ax==None or fig==None:
        plot=True
    else:
        count=False
        plot=False

    if norm==True:
        normalize_sw(spectral_width)

    #count i.e. station count plot
    if count == True:
               
        fig, ax = plt.subplots(2, constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})
        plot_sw(fig, ax[0], times, frequencies, spectral_width, log)
       
        days = np.arange(startdateplot, enddateplot, np.timedelta64(winlenhr, 'h'))
        ax[1].bar(days, statcount_all, width=1.0/(24/winlenhr), align='edge')
        ax[1].xaxis_date()
        ax[0].set_xlim(startdateplot, enddateplot)
        ax[1].set_xlim(startdateplot, enddateplot)
    else:
        if plot==True:
            fig, ax = plt.subplots()
        plot_sw(fig, ax, times, frequencies, spectral_width, log)

    if plot==True:
        plt.show()


def getSpectralWidthData(directory, startdate, enddate, count, winlenhr=6):

    currentdate = startdate
    numdays = int((enddate - currentdate)/86400)+1 #+1 to include enddate

    times_all = np.empty(numdays, dtype=object)
    sw_all = np.empty(numdays, dtype=object)
    statcount_all = np.empty(numdays,dtype=object)
    for i in range(numdays):
        try: 
            times, frequencies, spectral_width, statcount = readCovOutput(directory, currentdate, count)
        except:
            times = np.zeros(times.shape)
            spectral_width = np.empty(spectral_width.shape)
            spectral_width[:] = np.nan
            statcount = np.zeros(int(24/winlenhr))

        currdatearr = fillArray(dates.date2num(currentdate), len(times))
        times_all[i] = currdatearr + times/86400.0

        sw_all[i] = spectral_width
        statcount_all[i] = statcount
         
        #if appending new set of times, remove last time of previous entry (as corresponds to first of new)
        if i>0:             
            times_all[i-1] = times_all[i-1][:-1] 
        
        #increment date
        currentdate = currentdate + 86400 

    #stack arrays horizontally
    times_all = np.hstack(times_all)
    spectral_width_all = np.hstack(sw_all)
    statcount_all = np.hstack(statcount_all)
 
    return times_all, frequencies, spectral_width_all, statcount_all

def plot_sw(fig, ax, times, frequencies, spectral_width, log, ymax=10):

    img = ax.pcolormesh(times, frequencies, spectral_width, rasterized=True, cmap="viridis_r", shading='auto')
    ax.set_ylim([0.01, ymax])
    
    if log==True:
        ax.set_yscale('log')

    fig.colorbar(img, ax=ax).set_label("Covariance matrix spectral width")

    ax.xaxis_date()
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Hz)")

def normalize_sw(spectral_width):

   row, col = spectral_width.shape
   for i in range(col):
        spectral_width[:,i] = spectral_width[:,i] / np.max(spectral_width[:,i])

def getDayWaveform(datapath, channel, date, stations):

    st_jday = date.julday
    st_year = date.year
    
    st = csn.arraystream.ArrayStream()

    if channel == '*':
        channel = ''

    for root, dirs, files in os.walk(datapath+'/'+str(st_year)):        
        for file in files: 
            if file.endswith(channel+'.D.'+str(st_year)+'.'+str(st_jday)):
                if len(stations) == 0:
                    st += read(root+'/'+file)
                else:
                    for stat in stations:
                        if file.find(stat) != -1:
                            st += read(root+'/'+file)
       
    #print(st)  
    st.merge(fill_value=0)   
    #print(st)

    return st   

def readCovOutput(directory, date, statcount):
        
    filename = str(date.year)+'_'+str(date.julday)+'.npy'
    covresult = np.load('outputs/'+directory+'/'+filename, allow_pickle=True)
 
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


