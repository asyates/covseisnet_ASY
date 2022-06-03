#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from usefulFuncs import *
from covseisnetfunc import *
from obspy import UTCDateTime
from obspy import Stream
from obspy import read
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import signal
from scipy.signal import hilbert
from math import comb
import itertools

workdir = '/home/yatesal/covseisnet_ASY'

#plot SNR of CCFs    
def plotSNR(CCFparams, startdate, enddate, minlagwin, maxlagwin, fig=None, ax=None, vmin=np.nan, vmax=np.nan, stacksuffix='',norm=False):
    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
   
    #get filter low and high values:
    filtlowhigh, centfreqs = getFilters()

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
    asym_freq_array = np.empty((len(centfreqs), len(ccfdates)))
  
    #create lag time array
    samprate = 1.0/fs
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
    
    #set minimum and maximum index for snr windows
    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
    #maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
    #maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()

    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    #for each day in date range:
    for d in range(len(ccfdates)):
        
        day = convertDatetime64ToStr(ccfdates[d])
        
        #for each filter
        snrArray = np.zeros(len(filtlowhigh))

        for f in range(len(filtlowhigh)):

            #get stack corresponding to stacksize for given day, and also array of 1-day ccfs
            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component, stacksuffix=stacksuffix) 
                   
            #check if array (i.e. not nan)
            if isinstance(stack, (list, tuple, np.ndarray)):
                
                #calculate snr of ccfs
                period = 1.0/centfreqs[f]
                snr, ampenv, noise = compute_ccf_snr(ccfarray, fs, smooth_win=period, norm=norm)

                if maxlagwin == None:
                    #print(period)
                    maxlagwin0 = int(minlagwin + period*10)
                else:
                    maxlagwin0 = maxlagwin

                #print(centfreqs[f], minlagwin, maxlagwin0, maxlagwin0-minlagwin)

                maxidx_psnr = np.abs(lagtimes-maxlagwin0).argmin()
                maxidx_nsnr = np.abs(lagtimes-maxlagwin0*-1).argmin()

                #get snr values within SNR window (positive and negative)
                snr_p = snr[minidx_psnr:maxidx_psnr+1]
                snr_n = snr[maxidx_nsnr:minidx_nsnr+1]
                
                #average both negative and positive lag time snr
                avgSNR = np.mean([snr_p, snr_n]) 

                #append result for individual filter
                snrArray[f] = avgSNR

            else:
                snrArray[f] = np.nan
           
        #append result for each day
        snr_freq_array[:,d] = snrArray

    #cmap='Spectral'
    cmap='RdYlGn'
    #plot, also determining if vmin or vmax has been set
    if np.isnan(vmin) and np.isnan(vmax):
        #print(np.max(snr_freq_array[:,:-1]))

        #vmin=0
        vmax = np.nanquantile(snr_freq_array[:,:-1],0.999)

        img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmax=vmax, cmap=cmap, shading='auto')
    elif np.isnan(vmin):
        img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmax=vmax, cmap=cmap, shading='auto')
    elif np.isnan(vmax):
        img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmin=vmin, cmap=cmap, shading='auto')
    else:
        img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')

    #fig.colorbar(img, ax=ax).set_label('SNR  ('+str(minlagwin)+'-'+str(maxlagwin)+'s lag)')
    fig.colorbar(img, ax=ax).set_label('SNR')

    #cb_ax = fig.axes[1]
    #cb_ax.tick_params(labelsize=32)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)') 

    if plot == True:
        plt.show()

def getFilters():
    
    #define central frequencies
    centfreqs = [0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    filtlowhigh = np.empty(len(centfreqs), dtype=object)
    
    #compute upper and lower filter bands based on central frequency
    for f in range(len(centfreqs)):
        #flow = centfreqs[f]/(pow(pow(2,0.5),1.0/3.0))
        #fhigh = centfreqs[f]*(pow(pow(2,0.5),1.0/3.0))

        flow = centfreqs[f]/pow(2,0.5) 
        fhigh = centfreqs[f]*pow(2,0.5)

        filtlowhigh[f] = [flow, fhigh]

    return filtlowhigh, centfreqs

def plotAmpAsymmetry(CCFparams, startdate, enddate, fig=None, ax=None, stacksuffix='', minlag = 5):

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
   
    #get filter low and high values:
    filtlowhigh, centfreqs = getFilters()

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    asym_freq_array = np.empty((len(centfreqs), len(ccfdates)))
   
    #create lag time array
    samprate = 1.0/fs
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)

    #set index for getting amplitude each side of ccf
    minlag_amp= minlag
    posidxamp = np.abs(lagtimes-minlag_amp).argmin()
    negidxamp = np.abs(lagtimes-minlag_amp*-1).argmin()

    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    #for each day in date range:
    for d in range(len(ccfdates)):
        
        asymArray = np.zeros(len(filtlowhigh))
        day = convertDatetime64ToStr(ccfdates[d])

        for f in range(len(filtlowhigh)):
            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component, stacksuffix=stacksuffix)
 
            #check if array (i.e. not nan)
            if isinstance(stack, (list, tuple, np.ndarray)):

                #get maximum ampliude on each side of ccf, after the minimum lag time defined
                posamp = np.max(stack[posidxamp+1:])
                negamp = np.max(stack[:negidxamp])
                
                #compute ratio to log base 2
                asym_amp = np.log2(posamp/negamp)

                #append result for individual filter
                asymArray[f] = asym_amp

            else:
                asymArray[f] = np.nan

        #append result for each day
        asym_freq_array[:,d] = asymArray

    #get maximum value of asymmetry
    maxasym = np.nanmax(np.abs(asym_freq_array)) 
    asymLabel = 'Amp. Ratio (log$_{2}$)'

    img = ax.pcolormesh(ccfdates, centfreqs, asym_freq_array[:,:-1], rasterized=True, cmap="seismic", shading='auto', vmin=maxasym*-1, vmax=maxasym)
    fig.colorbar(img, ax=ax).set_label(asymLabel)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)')           

    if plot == True:
        plt.show()


def plotPhaseStack(CCFparams, startdate, enddate, minlagwin, maxlagwin, fig=None, ax=None, stacksuffix=''):
    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
   
    #get filter low and high values:
    filtlowhigh, centfreqs = getFilters()
    #filtindexes, centfreqs = getMSNoiseFilters()


    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
  
    #create lag time array
    samprate = 1.0/fs
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
    
    #set minimum and maximum index for snr windows
    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
    #maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
    #maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()

    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False


    #for each day in date range:
    for d in range(len(ccfdates)):
        
        day = convertDatetime64ToStr(ccfdates[d])
        
        #for each filter
        snrArray = np.zeros(len(filtlowhigh))

        for f in range(len(filtlowhigh)):
            
            
            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component, stacksuffix=stacksuffix, filt='01')
             
            #check if array (i.e. not nan)
            if isinstance(stack, (list, tuple, np.ndarray)):

                period=1.0/centfreqs[f]
                phasestack = compute_PhaseStack(ccfarray, fs, smooth_win=period) 

                if maxlagwin == None:
                    #print(period)
                    maxlagwin0 = int(minlagwin + period*10)
                else:
                    maxlagwin0 = maxlagwin

                maxidx_psnr = np.abs(lagtimes-maxlagwin0).argmin()
                maxidx_nsnr = np.abs(lagtimes-maxlagwin0*-1).argmin()

                #get snr values within SNR window (positive and negative)
                snr_p = phasestack[minidx_psnr:maxidx_psnr+1]
                snr_n = phasestack[maxidx_nsnr:minidx_nsnr+1]
 
                #average both negative and positive lag time snr
                avgSNR = np.mean([snr_p, snr_n]) 

                #append result for individual filter
                snrArray[f] = avgSNR

            else:
                snrArray[f] = np.nan

        #append result for each day
        snr_freq_array[:,d] = snrArray
    
    #vmin=0.25
    vmax = np.nanquantile(snr_freq_array[:,:-1],0.999)
    
    cmap='RdYlGn' 
    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, cmap=cmap, shading='auto', vmax=vmax)
    #fig.colorbar(img, ax=ax).set_label('PhSyn betw stacks  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
    fig.colorbar(img, ax=ax).set_label('Phase Stack Amp')

    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)') 

    if plot==True:
        plt.show()


def plotStackCC(CCFparams, startdate, enddate, minlagwin, maxlagwin, maxstack=5, fig=None, ax=None, stacksuffix=''):

    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
   
    #get filter low and high values:
    filtlowhigh, centfreqs = getFilters()
    

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
  
    #create lag time array
    samprate = 1.0/fs
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
    
    #set minimum and maximum index for snr windows
    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
    
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    #set CC threshold
    cc_thres = 0.99
    
    #for each day in date range:
    for d in range(len(ccfdates)):
        
        day = convertDatetime64ToStr(ccfdates[d])
        
        #for each filter
        snrArray = np.zeros(len(filtlowhigh))

        for f in range(len(filtlowhigh)):

            period=1.0/centfreqs[f]

            if maxlagwin == None:
                #print(period)
                maxlagwin0 = int(minlagwin + period*15)
            else:
                maxlagwin0 = maxlagwin

            maxidx_psnr = np.abs(lagtimes-maxlagwin0).argmin()
            maxidx_nsnr = np.abs(lagtimes-maxlagwin0*-1).argmin()

            cc_curr = 0
            ccf_count = 1

            #get stack of max stacksize
            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, maxstack, day, filtlowhigh[f], fs, loc=loc, component=component, stacksuffix=stacksuffix, filt='01') 
           
            #slice individual arrays to be just times of interest
            ccfarray_p = [ccfarray[i][minidx_psnr:maxidx_psnr+1] for i in range(len(ccfarray))]
            ccfarray_n = [ccfarray[i][maxidx_nsnr:minidx_nsnr+1] for i in range(len(ccfarray))]

            #recombine 
            ccfarray_new = np.concatenate((ccfarray_n, ccfarray_p), axis = 1)


            while cc_curr < cc_thres:

                ccf_count += 1 #increment count

                #get stacksize+1 because will compare two successive stacks
                #stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, ccf_count, day, filtlowhigh[f], fs, loc=loc, component=component, stacksuffix=stacksuffix, filt='01') 

                stcf0 = np.mean(ccfarray_new[(-1*ccf_count):], axis=0)
                stcf1 = np.mean(ccfarray_new[(-1*(ccf_count-1)):], axis=0)                 

                #get values of stcf0 within min and max lag time (positive and negative)
                #stcf0_p = stcf0[minidx_psnr:maxidx_psnr+1]
                #stcf0_n = stcf0[maxidx_nsnr:minidx_nsnr+1]
                #stcf0_new = np.concatenate((stcf0_n, stcf0_p), axis=None)

                #get values of stcf0 within min and max lag time (positive and negative)
                #stcf1_p = stcf1[minidx_psnr:maxidx_psnr+1]
                #stcf1_n = stcf1[maxidx_nsnr:minidx_nsnr+1]
                #stcf1_new = np.concatenate((stcf1_n, stcf1_p), axis=None)

                cc_curr = np.corrcoef(stcf0, stcf1)[1,0]
                
                if ccf_count >= maxstack:
                    break


            #append result for individual filter
            snrArray[f] = ccf_count

            print(f, day)

        #append result for each day
        snr_freq_array[:,d] = snrArray



    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmin=2, cmap="RdYlGn_r", shading='auto')
    #fig.colorbar(img, ax=ax).set_label('PhSyn betw stacks  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
    fig.colorbar(img, ax=ax).set_label('Stacksize (CC = '+str(cc_thres)+')')

    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)')

    if plot==True:
        plt.show()



def getCCFStack(directory, network, stat1, stat2, stacksize, enddate, frange, fs, filt='01', component='ZZ', loc='00', stacksuffix=''):
    #return stack of 1-day ccfs corresponding to stacksize and enddate, and also the individual 1-day CCFs (after filtering)

    #directory containing stacks used in SNR computation
    stackdirpath = directory+'/STACKS'+stacksuffix+'/'+filt+'/001_DAYS/'+component+'/'

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = enddate_dt - timedelta(days=stacksize-1)
        
    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    
    #read 1-day CCFs
    st = Stream()
    statpair = sorted((stat1, stat2))
    statdir = network+'.'+statpair[0]+'.'+loc+'_'+network+'.'+statpair[1]+'.'+loc+'/'

    #error booleans
    errorData = False
    errorFS = False
    
    for ccfdate in ccfdates: 
        fname = convertDatetime64ToStr(ccfdate)+'.MSEED'
        fullpath = stackdirpath+statdir+fname
        if os.path.isfile(fullpath):
            st += read(fullpath) 
        else:
            print(fullpath + ' missing')

    # print(st)

    if len(st) < stacksize:
       errorData = True

    if len(st) != 0:
                   
        if frange != [0.0, 0.0]:
            st.taper(0.05)
            st.filter("bandpass", freqmin=frange[0], freqmax=frange[1], zerophase=True, corners=4)

        sampfreq = st[0].stats.sampling_rate
        samprate = 1.0/sampfreq

        if sampfreq != fs:
            errorFS = True
            
        #create array of 1-day ccfs (moving from stream object to numpy array)
        ccfs_1day = np.empty(len(st), dtype=object)
        for i in range(len(st)):
            ccfs_1day[i] = st[i].data
        
        #compute linear stack to return
        stack = np.mean(ccfs_1day, axis=0)
    else:
        stack = np.nan
        ccfs_1day = np.nan
    
    return stack, ccfs_1day

def compute_ccf_snr(ccfarray, fs, smooth_win = 10, norm=False):

    #normalize ccfs
    if norm==True:
        #get max for each individual ccf
        ccfs_max = np.max(np.abs(ccfarray.tolist()), axis=1)
        ccfarray = ccfarray / ccfs_max

    #calculate mean of these, equivalent to the N-day stack
    ccf_mean = np.mean(ccfarray, axis=0)
    
    #define taper and taper signal for performing hilbert transform
    taper = signal.tukey(len(ccf_mean), alpha=0.2)
    ccf_mean_tapered = ccf_mean * taper

    #compute amplitude envelope
    analytic_signal = hilbert(ccf_mean_tapered)
    ampenv = np.abs(analytic_signal)
    
    #smooth with hanning window
    ampenv_smoothed = smooth(ampenv, window_len = int(fs*smooth_win))
    if len(ampenv_smoothed) != len(ccf_mean):
        ampenv_smoothed = signal.resample(ampenv_smoothed, len(ccf_mean))

    #compute noise signal, following Clarke et al. (2011):
    ccfarray_squared = np.square(ccfarray.tolist())      
    avg_squared = np.mean(ccfarray_squared,axis=0) #take avg of squares
    squared_avg = np.square(ccf_mean) #square the average

    noise = np.sqrt((avg_squared-squared_avg)/(len(ccfarray)-1))
    
    #smooth noise signal (with same length as earlier smooth)
    noise_smoothed = smooth(noise, window_len = int(fs*smooth_win))
    if len(noise_smoothed) != len(noise):
        noise_smoothed = signal.resample(noise_smoothed, len(noise))

    #compute SNR
    snr = ampenv_smoothed / noise_smoothed
    
    return snr, ampenv_smoothed, noise_smoothed  

def compute_PhaseStack(ccfarray, fs, smooth_win = 10, median=True):

    #define taper prior to normalising each ccf by its amplitude envelope
    taper_perc = 0.1 
    taper = signal.tukey(len(ccfarray[0]), alpha=taper_perc)

    #array for phase angles
    ccfs_norm = np.empty(len(ccfarray), dtype=object)

    for c in range(len(ccfs_norm)):
        
        #taper pre-hilbert transform
        ccf_tapered = ccfarray[c] * taper
        
        #get phase angle
        analytic_signal = hilbert(ccf_tapered)
        angle = np.angle(analytic_signal)
        ccfnorm = np.exp(1j*angle)
        ccfs_norm[c] = ccfnorm

    if len(ccfs_norm) > 0:
        
        ampenv = np.abs(np.sum(ccfs_norm.tolist(), axis=0))/len(ccfs_norm)

        #smooth with hanning window
        ampenv_smoothed = smooth(ampenv, window_len = fs*smooth_win)
        if len(ampenv_smoothed) != len(ccfnorm):
            ampenv_smoothed = signal.resample(ampenv_smoothed, len(ccfnorm))
    else:
        ampenv_smoothed = np.nan

    return ampenv_smoothed

def plotInterferogram(CCFparams, startdate, enddate, frange, fig=None, ax=None, stacksuffix='', norm=False):

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)

    #create lag time array
    samprate = 1.0/fs
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    ccf_array = np.empty((len(lagtimes),len(ccfdates)))

    #calculate SNR for each day
    for d in range(len(ccfdates)):
        day = convertDatetime64ToStr(ccfdates[d])

        #get stack corresponding to stacksize for given day, and also array of 1-day ccfs
        stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, frange, fs, loc=loc, component=component, stacksuffix=stacksuffix) 
        
        if norm == True:
            stack = stack / np.max(np.abs(stack))

        ccf_array[:,d] = stack

    #define the 99% percentile of data for visualisation purposes
    clim = np.nanpercentile(ccf_array, 99)

    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(12,8))
        plot = True
    else:
        plot = False

    img = ax.pcolormesh(ccfdates, lagtimes, ccf_array[:,:-1], vmin=-clim, vmax=clim, rasterized=True, cmap='seismic')
    fig.colorbar(img, ax=ax).set_label('')
    
    #plt.colorbar()
    ax.set_title('Interferogram')
    ax.set_ylabel('Lag Time (s)')
    ax.set_xlim(np.datetime64(startdate), np.datetime64(enddate))

    if plot == True:
        plt.show()



def plotSNR_time(CCFparams, startdate, enddate, frange, filt='01', stacksuffix='', norm=False, fig=None, ax=None):

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
        
    #create lag times array
    samprate = 1.0/fs
    x = np.arange(-1*maxlag, maxlag+samprate, samprate)
    #minlag=5
    #posidx = np.abs(x-minlag).argmin()
    #negidx = np.abs(x-minlag*-1).argmin()

    centfreq = (frange[0]+frange[1])/2
    #centfreq = frange[0]
    #print(centfreq)

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_array = np.empty((len(x),len(ccfdates)))
    #asym_array_amp = np.zeros(len(ccfdates))
    #asym_array_snr = np.zeros(len(ccfdates))
    
    #calculate SNR for each day
    for d in range(len(ccfdates)):
        day = convertDatetime64ToStr(ccfdates[d])

        #get stack corresponding to stacksize for given day, and also array of 1-day ccfs
        stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, frange, fs, loc=loc, component=component, stacksuffix=stacksuffix, filt=filt) 
        
        #calculate snr of ccfs
        period = 1.0/centfreq
        snr, ampenv, noise = compute_ccf_snr(ccfarray, fs, smooth_win=period, norm=norm)

        snr_array[:,d] = snr
        
        #calculate asymmetry
        #posamp = np.max(stack[posidx+1:])
        #negamp = np.max(stack[:negidx])

        #possnr = np.max(snr[posidx+1:])
        #negsnr = np.max(snr[:negidx])

        #asym_array_amp[d] = np.log2(posamp/negamp)
        #asym_array_snr[d] = np.log2(possnr/negsnr)

    #plot SNR colormap
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    #title = network+'_'+stat1+':'+network+'_'+stat2+'('+component+', f'+filt+str(frange)+', m'+str(stacksize)+')'
    #fig.suptitle(title)

    #choose max snr for plot
    #vmax = np.nanmax(snr_array)
    #print(vmax)
    #if vmax > 20:
    #    vmax=20
    #vmax=10

    vmax = np.percentile(snr_array,99)

    img = ax.pcolormesh(ccfdates, x, snr_array[:,:-1], rasterized=True, vmax=vmax, cmap="Spectral_r", shading='auto')
    #fig.colorbar(img, ax=ax[0])
    fig.colorbar(img, ax=ax).set_label("SNR")
    ax.set_ylabel('lag time (s)')
    ax.set_xlim(startdateplot,enddateplot)

    #ax[1].plot(ccfdates,asym_array_amp, color='black', label='amplitude')
    #ax[1].plot(ccfdates,asym_array_snr, color='red', label='snr')
    #ax[1].set_xlim(startdateplot,enddateplot)
    #ax[1].set_ylabel('log2(pos/neg)')
    #ax[1].legend() 

    if plot == True:
        plt.show()

def plotPhaseStack_time(CCFparams, startdate, enddate, frange, filt='01', stacksuffix='', norm=False, fig=None, ax=None):

    #reassign variables
    noisedir = CCFparams[0]
    network = CCFparams[1]
    loc = CCFparams[2]
    stat1 = CCFparams[3]
    stat2 = CCFparams[4]
    component = CCFparams[5]
    stacksize = CCFparams[6]
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
        
    #create lag times array
    samprate = 1.0/fs
    x = np.arange(-1*maxlag, maxlag+samprate, samprate)
    #print(len(x))
    #minlag=5
    #posidx = np.abs(x-minlag).argmin()
    #negidx = np.abs(x-minlag*-1).argmin()

    centfreq = (frange[0]+frange[1])/2
    centfreq= frange[0]

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_array = np.empty((len(x),len(ccfdates)))
    #asym_array_amp = np.zeros(len(ccfdates))
    #asym_array_snr = np.zeros(len(ccfdates))
    
    #calculate SNR for each day
    for d in range(len(ccfdates)):
        day = convertDatetime64ToStr(ccfdates[d])

        #get stack corresponding to stacksize for given day, and also array of 1-day ccfs
        stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, frange, fs, loc=loc, component=component, stacksuffix=stacksuffix, filt=filt) 
        

        #check if array (i.e. not nan)
        if isinstance(stack, (list, tuple, np.ndarray)):

        #calculate snr of ccfs
            period = 1.0/centfreq
            phasestack = compute_PhaseStack(ccfarray, fs, smooth_win=period)
        
            snr_array[:,d] = phasestack
        
        else:
            arr = np.empty(len(x))
            arr[:] = np.nan
            snr_array[:,d] = arr

        #calculate asymmetry
        #posamp = np.max(stack[posidx+1:])
        #negamp = np.max(stack[:negidx])

        #possnr = np.max(snr[posidx+1:])
        #negsnr = np.max(snr[:negidx])

        #asym_array_amp[d] = np.log2(posamp/negamp)
        #asym_array_snr[d] = np.log2(possnr/negsnr)

    #plot SNR colormap
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    #title = network+'_'+stat1+':'+network+'_'+stat2+'('+component+', f'+filt+str(frange)+', m'+str(stacksize)+')'
    #fig.suptitle(title)

    #choose max snr for plot
    #vmax = np.nanmax(snr_array)
    #print(vmax)
    #if vmax > 20:
    #    vmax=20
    #vmax=10

    #vmax = np.percentile(snr_array,95)

    img = ax.pcolormesh(ccfdates, x, snr_array[:,:-1], rasterized=True, cmap="Spectral_r", shading='auto')
    #fig.colorbar(img, ax=ax[0])
    fig.colorbar(img, ax=ax).set_label("Phase Stack")
    ax.set_ylabel('lag time (s)')
    ax.set_xlim(startdateplot,enddateplot)

    #ax[1].plot(ccfdates,asym_array_amp, color='black', label='amplitude')
    #ax[1].plot(ccfdates,asym_array_snr, color='red', label='snr')
    #ax[1].set_xlim(startdateplot,enddateplot)
    #ax[1].set_ylabel('log2(pos/neg)')
    #ax[1].legend() 

    if plot == True:
        plt.show()



#plot spectrogram saved in .csv format (based on other codes not included)
def plot_spectogram(inputfile, startdate, enddate, classic=True, demean=False,norm=False, Cor_norm=False, vmax=5000, fig=None, ax=None):

    df=pd.read_csv(inputfile,parse_dates=True,index_col=0,header=0)
    #df=df.resample('1T').mean()

    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(11,4))
        plot=True
    else:
        plot=False

    if norm:
        row_max = df.max(axis=1)
        row_min = df.min(axis=1)
        df = (df -row_min[:, np.newaxis]) / (row_max[:, np.newaxis]-row_min[:, np.newaxis])
        
    if Cor_norm:
        df = (df-df.min())/(df.max()-df.min())
        
    if demean:
        df = df-df.mean(axis=0)

    df.columns= pd.Float64Index(df.columns)

    # si norm
    if norm:
        mesh=ax.pcolormesh(df.index,df.columns,df.T,vmin=0.0,vmax=1.0,cmap='Spectral_r')
    if Cor_norm:
        mesh=ax.pcolormesh(df.index,df.columns,df.T,vmin=0.0,vmax=0.4,cmap='Spectral_r')
    if classic:
        mesh=ax.pcolormesh(df.index,df.columns,df.T,vmin=-120,vmax=vmax,cmap='Spectral_r')
    if demean:
        mesh=ax.pcolormesh(df.index,df.columns,df.T,vmin=0,vmax=2e5,cmap='Spectral_r')

    fig.colorbar(mesh, ax=ax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    #ax.set_title(inputfile)
    ax.set_xlim(pd.Timestamp(startdate), pd.Timestamp(enddate))
 
    if plot == True:
        plt.show()

def get_interstation_distance(net1, stat1, net2, stat2):

    df = pd.read_csv(workdir+'/'+'station_info.csv', header=0)
    
    #print(stat2, net2) 
    stat1_info = df.loc[(df['station'] == stat1) & (df['network'] == net1)]
    stat2_info = df.loc[(df['station'] == stat2) & (df['network'] == net2)]
    
    #print(stat1_info)
    #print(stat2_info)

    lat1 = stat1_info['latitude'].values[0]
    lon1 = stat1_info['longitude'].values[0]
    lat2 = stat2_info['latitude'].values[0]
    lon2 = stat2_info['longitude'].values[0]

    distance = getDistance(lat1,lon1,lat2,lon2)

    return distance

#Function no longer used (was used while multiple filters were being defined in MSNoise, prior to filtering using broadband CCF 
#def getMSNoiseFilters():
#
#    filtindexes = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
#
#    filtlowhigh = [[0.01,0.1],[0.03,0.13],[0.05,0.15],[0.1,0.2],[0.15,0.25],[0.2,0.3],[0.25,0.35],[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9],[0.9,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6],[1.6,1.8],[1.8,2.0],[2.0,2.5],[2.5,3.0],[3.0,3.5],[3.5,4.0],[4.0,5.0],[5.0,6.0],[6.0,8.0],[8.0,10.0]]
#
#    centfreqs = np.zeros(len(filtindexes))
#    for f in range(len(filtlowhigh)):
#        cf = np.mean(filtlowhigh[f])
#        centfreqs[f] = cf
#
#    return filtindexes, centfreqs

#def compute_chi(directory, network, stat1, stat2, stacksize, enddate, frange, fs, minlagwin, maxlagwin, filt='01', component='ZZ', smooth_win = 10, loc='00', maxlag=120, norm=False):
#
#    #directory containing stacks used in SNR computation
#    stackdirpath = directory+'/STACKS/'+filt+'/001_DAYS/'+component+'/'
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = enddate_dt - timedelta(days=stacksize-1)
#        
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    
#    #read 1-day CCFs
#    st = Stream()
#    statpair = sorted((stat1, stat2))
#    statdir = network+'.'+statpair[0]+'.'+loc+'_'+network+'.'+statpair[1]+'.'+loc+'/'
#
#    #error booleans
#    errorData = False
#    errorFS = False
#    
#    for ccfdate in ccfdates: 
#        fname = convertDatetime64ToStr(ccfdate)+'.MSEED'
#        fullpath = stackdirpath+statdir+fname
#        if os.path.isfile(fullpath):
#            st += read(fullpath) 
#
#    # print(st)
#
#    if len(st) < stacksize:
#       errorData = True
#
#    if len(st) != 0:
#                   
#        st.taper(0.05)
#        st.filter("bandpass", freqmin=frange[0], freqmax=frange[1], zerophase=True, corners=4)
#
#        sampfreq = st[0].stats.sampling_rate
#        samprate = 1.0/sampfreq
#
#        if sampfreq != fs:
#            errorFS = True
#            
#        #create array of 1-day ccfs (moving from stream object to numpy array)
#        ccfs_1day = np.empty(len(st), dtype=object)
#        for i in range(len(st)):
#            ccfs_1day[i] = st[i].data
#        
#        #compute linear stack to return
#        stack = np.mean(ccfs_1day, axis=0)
#
#        #normalize ccfs
#        if norm==True:
#            #get max for each individual ccf
#            ccfs_max = np.max(np.abs(ccfs_1day.tolist()), axis=1)
#            ccfs_1day = ccfs_1day / ccfs_max
#
#        N = len(ccfs_1day)
#     
#        #sum the different stacks (stcf1 being the stack with the latest CCF, stcf0 being without)
#        stcf0 = np.sum(ccfs_1day[0:N-1], axis=0)
#        stcf1 = np.sum(ccfs_1day[0:N], axis=0)        
#
#        #create lag time array
#        samprate = 1.0/fs
#        lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#
#        #get index corresponding to minimum and maximum lag time each side of CCF
#        minidx_pchi = np.abs(lagtimes-minlagwin).argmin()
#        minidx_nchi = np.abs(lagtimes-minlagwin*-1).argmin()
#        maxidx_pchi = np.abs(lagtimes-maxlagwin).argmin()
#        maxidx_nchi = np.abs(lagtimes-maxlagwin*-1).argmin()
#
#        #for stcf0, get negative and positive lag times, and make new array (normalise both individually)
#        stcf0_neg = stcf0[maxidx_nchi:minidx_nchi+1]
#        stcf0_pos = stcf0[minidx_pchi:maxidx_pchi+1]
#        stcf0_neg = stcf0_neg / np.max(np.abs(stcf0_neg))
#        stcf0_pos = stcf0_pos / np.max(np.abs(stcf0_pos))
#        stcf0_new = np.array([stcf0_neg, stcf0_pos])
#        stcf0_new = np.hstack(stcf0_new)  
#
#        #for stcf1, get negative and positive lag times, and make new array (normalise both individually)
#        stcf1_neg = stcf1[maxidx_nchi:minidx_nchi+1]
#        stcf1_pos = stcf1[minidx_pchi:maxidx_pchi+1]
#        stcf1_neg = stcf1_neg / np.max(np.abs(stcf1_neg))
#        stcf1_pos = stcf1_pos / np.max(np.abs(stcf1_pos))
#        stcf1_new = np.array([stcf1_neg, stcf1_pos])
#        stcf1_new = np.hstack(stcf1_new)
#
#        #take the squared difference between the stacks, both normalized by the maximum amplitude of each
#        diffsq = np.square(stcf1_new - stcf0_new)
#        
#        #sum the squared differences
#        summ = np.sum(diffsq)
#
#        #take the squareroot of the sum
#        chi = np.sqrt(summ)
#
#        return chi, [errorData, errorFS]

#def plotAmpEnv(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax):
#    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   
#
#    #reassign variables
#    noisedir = CCFparams[0]
#    network = CCFparams[1]
#    loc = CCFparams[2]
#    stat1 = CCFparams[3]
#    stat2 = CCFparams[4]
#    component = CCFparams[5]
#    stacksize = CCFparams[6]
#    fs = CCFparams[7]
#    maxlag = CCFparams[8]
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = convertDateStrToDatetime(startdate)
#    startdateplot = np.datetime64(startdate)
#    enddateplot = np.datetime64(enddate)
#   
#    #get filter low and high values:
#    filtlowhigh, centfreqs = getFilters()
#
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
#  
#    #create lag time array
#    samprate = 1.0/fs
#    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#    
#    #set minimum and maximum index for snr windows
#    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
#    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
#    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
#    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()
#
#    #for each day in date range:
#    for d in range(len(ccfdates)):
#        
#        day = convertDatetime64ToStr(ccfdates[d])
#        
#        #for each filter
#        snrArray = np.zeros(len(filtlowhigh))
#
#        for f in range(len(filtlowhigh)):
#
#            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component)
#
#            #get amplitude envelope of normalised CCFs 
#            period = 1.0/centfreqs[f]
#            ampenv = compute_AmpEnv(ccfarray, fs, smooth_win=period)
#        
#            #check if array (i.e. not nan)
#            if isinstance(ampenv, (list, tuple, np.ndarray)):
#
#                #get snr values within SNR window (positive and negative)
#                snr_p = ampenv[minidx_psnr:maxidx_psnr+1]
#                snr_n = ampenv[maxidx_nsnr:minidx_nsnr+1]
# 
#                #average both negative and positive lag time snr
#                avgSNR = np.mean([snr_p, snr_n]) 
#
#                #append result for individual filter
#                snrArray[f] = avgSNR
#
#            else:
#                snrArray[f] = np.nan
#
#        #append result for each day
#        snr_freq_array[:,d] = snrArray
#   
#    #vmin = np.min(snr_freq_array[:,:-1])
#    vmin=0.1
#    vmax = np.max(snr_freq_array[:,:-1])
#
#    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, cmap="jet", shading='auto', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
#    #fig.colorbar(img, ax=ax).set_label('ampenv  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
#    fig.colorbar(img, ax=ax).set_label('ampenv norm')
#
#    ax.set_yscale('log')
#    ax.set_ylabel('Frequency (Hz)') 

#def plotNormAmpEnv(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax, axstart=0):
#    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   
#
#    #reassign variables
#    noisedir = CCFparams[0]
#    network = CCFparams[1]
#    loc = CCFparams[2]
#    stat1 = CCFparams[3]
#    stat2 = CCFparams[4]
#    component = CCFparams[5]
#    stacksize = CCFparams[6]
#    fs = CCFparams[7]
#    maxlag = CCFparams[8]
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = convertDateStrToDatetime(startdate)
#    startdateplot = np.datetime64(startdate)
#    enddateplot = np.datetime64(enddate)
#   
#    #get filter low and high values:
#    filtlowhigh, centfreqs = getFilters()
#
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
#  
#    #create lag time array
#    samprate = 1.0/fs
#    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#    
#    #set minimum and maximum index for snr windows
#    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
#    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
#    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
#    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()
#
#    #for each day in date range:
#    for d in range(len(ccfdates)):
#        
#        day = convertDatetime64ToStr(ccfdates[d])
#        
#        #for each filter
#        snrArray = np.zeros(len(filtlowhigh))
#
#        for f in range(len(filtlowhigh)):
#
#            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component)
#
#            #get amplitude envelope of normalised CCFs 
#            period = 1.0/centfreqs[f]
#            ampenv = compute_NormAmpEnv(ccfarray, fs, smooth_win = period)
#        
#            #check if array (i.e. not nan)
#            if isinstance(ampenv, (list, tuple, np.ndarray)):
#
#                #get snr values within SNR window (positive and negative)
#                snr_p = ampenv[minidx_psnr:maxidx_psnr+1]
#                snr_n = ampenv[maxidx_nsnr:minidx_nsnr+1]
# 
#                #average both negative and positive lag time snr
#                avgSNR = np.mean([snr_p, snr_n]) 
#
#                #append result for individual filter
#                snrArray[f] = avgSNR
#
#            else:
#                snrArray[f] = np.nan
#
#        #append result for each day
#        snr_freq_array[:,d] = snrArray
#    
#    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, cmap="jet", shading='auto')
#    #fig.colorbar(img, ax=ax).set_label('ampenv  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
#    fig.colorbar(img, ax=ax).set_label('ampenv norm')
#
#    ax.set_yscale('log')
#    ax.set_ylabel('Frequency (Hz)') 

#def plotAvgPhaseSync(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax, median=True):
#    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   
#
#    #reassign variables
#    noisedir = CCFparams[0]
#    network = CCFparams[1]
#    loc = CCFparams[2]
#    stat1 = CCFparams[3]
#    stat2 = CCFparams[4]
#    component = CCFparams[5]
#    stacksize = CCFparams[6]
#    fs = CCFparams[7]
#    maxlag = CCFparams[8]
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = convertDateStrToDatetime(startdate)
#    startdateplot = np.datetime64(startdate)
#    enddateplot = np.datetime64(enddate)
#   
#    #get filter low and high values:
#    filtlowhigh, centfreqs = getFilters()
#
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
#  
#    #create lag time array
#    samprate = 1.0/fs
#    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#    
#    #set minimum and maximum index for snr windows
#    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
#    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
#    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
#    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()
#
#    #for each day in date range:
#    for d in range(len(ccfdates)):
#        
#        day = convertDatetime64ToStr(ccfdates[d])
#        
#        #for each filter
#        snrArray = np.zeros(len(filtlowhigh))
#
#        for f in range(len(filtlowhigh)):
#
#            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, loc=loc, component=component)
#
#            #get amplitude envelope of normalised CCFs 
#            period = 1.0/centfreqs[f]
#            ampenv = compute_AvgPhaseSync(ccfarray, fs, median=median, smooth_win=period)
#        
#            #check if array (i.e. not nan)
#            if isinstance(ampenv, (list, tuple, np.ndarray)):
#
#                #get snr values within SNR window (positive and negative)
#                snr_p = ampenv[minidx_psnr:maxidx_psnr+1]
#                snr_n = ampenv[maxidx_nsnr:minidx_nsnr+1]
# 
#                #average both negative and positive lag time snr
#                avgSNR = np.mean([snr_p, snr_n]) 
#
#                #append result for individual filter
#                snrArray[f] = avgSNR
#
#            else:
#                snrArray[f] = np.nan
#
#        #append result for each day
#        snr_freq_array[:,d] = snrArray
#    
#    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, cmap="jet", shading='auto')
#    #fig.colorbar(img, ax=ax).set_label('Avg Phase Sync  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
#    fig.colorbar(img, ax=ax).set_label('AvgPS Med='+str(median))
#
#    ax.set_yscale('log')
#    ax.set_ylabel('Frequency (Hz)') 

#def plotPhaseSync(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax):
#    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   
#
#    #reassign variables
#    noisedir = CCFparams[0]
#    network = CCFparams[1]
#    loc = CCFparams[2]
#    stat1 = CCFparams[3]
#    stat2 = CCFparams[4]
#    component = CCFparams[5]
#    stacksize = CCFparams[6]
#    fs = CCFparams[7]
#    maxlag = CCFparams[8]
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = convertDateStrToDatetime(startdate)
#    startdateplot = np.datetime64(startdate)
#    enddateplot = np.datetime64(enddate)
#   
#    #get filter low and high values:
#    filtlowhigh, centfreqs = getFilters()
#    #filtindexes, centfreqs = getMSNoiseFilters()
#
#
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    snr_freq_array = np.empty((len(centfreqs),len(ccfdates)))
#  
#    #create lag time array
#    samprate = 1.0/fs
#    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#    
#    #set minimum and maximum index for snr windows
#    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
#    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
#    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
#    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()
#
#    #for each day in date range:
#    for d in range(len(ccfdates)):
#        
#        day = convertDatetime64ToStr(ccfdates[d])
#        
#        #for each filter
#        snrArray = np.zeros(len(filtlowhigh))
#
#        for f in range(len(filtlowhigh)):
#            
#            #get stacksize+1 because will compare two successive stacks
#            stack, ccfarray =  getCCFStack(noisedir, network, stat1, stat2, stacksize+1, day, filtlowhigh[f], fs, loc=loc, component=component, filt='01')
#            
#            period=1.0/centfreqs[f]
#            phasesync = compute_PhaseSync(ccfarray, fs, smooth_win=period) 
#
#            #check if array (i.e. not nan)
#            if isinstance(phasesync, (list, tuple, np.ndarray)):
#
#                #get snr values within SNR window (positive and negative)
#                snr_p = phasesync[minidx_psnr:maxidx_psnr+1]
#                snr_n = phasesync[maxidx_nsnr:minidx_nsnr+1]
# 
#                #average both negative and positive lag time snr
#                avgSNR = np.mean([snr_p, snr_n]) 
#
#                #append result for individual filter
#                snrArray[f] = avgSNR
#
#            else:
#                snrArray[f] = np.nan
#
#        #append result for each day
#        snr_freq_array[:,d] = snrArray
#    
#    img = ax.pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], rasterized=True, vmin=0.8, cmap="jet", shading='auto')
#    #fig.colorbar(img, ax=ax).set_label('PhSyn betw stacks  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
#    fig.colorbar(img, ax=ax).set_label('PhSyn Stacks')
#
#    ax.set_yscale('log')
#    ax.set_ylabel('Frequency (Hz)') 

#def plotChi(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax):
#    #Note that CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]   
#
#    #reassign variables
#    noisedir = CCFparams[0]
#    network = CCFparams[1]
#    loc = CCFparams[2]
#    stat1 = CCFparams[3]
#    stat2 = CCFparams[4]
#    component = CCFparams[5]
#    stacksize = CCFparams[6]
#    fs = CCFparams[7]
#    maxlag = CCFparams[8]
#
#    #convert dates to UTCDatetime and designate startdate
#    enddate_dt = convertDateStrToDatetime(enddate)
#    startdate_dt = convertDateStrToDatetime(startdate)
#    startdateplot = np.datetime64(startdate)
#    enddateplot = np.datetime64(enddate)
#   
#    #get filter low and high values:
#    filtlowhigh, centfreqs = getFilters()
#
#    #create date array and reading single day files
#    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
#    chi_freq_array = np.empty((len(centfreqs),len(ccfdates)))
#      
#    #create lag time array
#    samprate = 1.0/fs
#    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
#        
#    #for each day in date range:
#    for d in range(len(ccfdates)):
#        
#        #define booleans for errors that can occur
#        errorData = False
#        errorFS = False
#
#        day = convertDatetime64ToStr(ccfdates[d])
#        
#        #for each filter
#        chiArray = np.zeros(len(filtlowhigh))
#
#        for f in range(len(filtlowhigh)):
#
#            chi, errors =  compute_chi(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, minlagwin, maxlagwin, norm=False, loc=loc, component=component, filt='01')
#
#            if errors[0] == True:
#                errorData = True
#            if errors[1] == True:
#                errorFS = True
#            
#            chiArray[f] = chi
#
#        if errorData == True:
#            print("WARNING: Data missing for stack ending "+day)
#
#        if errorFS == True:
#            print("WARNING: YOU HAVE PROVIDED A SAMPLING FREQUENCY DIFFERENT TO THOSE THE CCFs")
#
#        #append result for each day
#        chi_freq_array[:,d] = chiArray
#        
#    vmin = chi_freq_array[:,:-1].min()
#    vmax = chi_freq_array[:,:-1].max()
#
#    img1 = ax.pcolormesh(ccfdates, centfreqs, chi_freq_array[:,:-1], rasterized=True, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="jet_r", shading='auto')
#    fig.colorbar(img1, ax=ax).set_label('Chi  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
#    ax.set_yscale('log')
#    ax.set_ylabel('Frequency (Hz)') 
 
#def compute_PhaseSync(ccfarray, fs, smooth_win = 10):
#
#    stack2 = np.mean(ccfarray[1:].tolist(), axis=0)
#    stack1 = np.mean(ccfarray[:-1].tolist(), axis=0)
#    #print(len(ccfarray[1:]), len(ccfarray[:-1]))
#
#    taper_perc = 0.05
#    taper = signal.tukey(len(stack2), alpha=taper_perc)
#
#    stack2 = stack2 * taper
#    stack1 = stack1 * taper
#
#    s2_analytic_signal = hilbert(stack2)
#    s2_angle = np.angle(s2_analytic_signal)
#
#    s1_analytic_signal = hilbert(stack1)
#    s1_angle = np.angle(s1_analytic_signal)
#
#    phasesync = 1 - np.sin(np.abs(s2_angle-s1_angle)/2)
#    
#    phasesync_smooth = smooth(phasesync, window_len=int(fs*smooth_win))
#    #return phasesync
#    return phasesync_smooth

#def compute_AmpEnv(ccfarray, fs, smooth_win = 10):
#
#    #normalise ccfs
#    ccfs_max = np.max(np.abs(ccfarray.tolist()), axis=1)
#    ccfs_norm = ccfarray / ccfs_max
#
#    if len(ccfs_norm) > 0:
#        #calculate mean of these, equivalent to the N-day linear stack
#        ccf_mean = np.mean(ccfs_norm, axis=0)
#    
#        #compute amplitude envelope
#        analytic_signal = hilbert(ccf_mean)
#        ampenv = np.abs(analytic_signal)
#        
#        #smooth with hanning window
#        ampenv_smoothed = smooth(ampenv, window_len = int(fs*smooth_win))
#    else:
#        ampenv_smoothed = np.nan
#
#    return ampenv_smoothed

#def compute_NormAmpEnv(ccfarray, fs, smooth_win = 10):
#
#    #define taper prior to normalising each ccf by its amplitude envelope
#    taper_perc = 0.1 
#    taper = signal.tukey(len(ccfarray[0]), alpha=taper_perc)
#
#    #array for normalised ccfs
#    ccfs_norm = np.empty(len(ccfarray), dtype=object)
#
#    for c in range(len(ccfs_norm)):
#        
#        #taper pre-hilbert transform
#        ccf_tapered = ccfarray[c] * taper
#         
#        #get amplitude envelope
#        analytic_signal = hilbert(ccf_tapered)
#        ampenv_1day = np.abs(analytic_signal)
#                
#        #normalise with amplitude envelope, and apply taper again to avoid large end values
#        ccfnorm = np.divide(ccfarray[c],ampenv_1day) * taper       
#        #ccfnorm = ccfarray[c] / (np.abs(ccfarray[c])+1e-10)
#        #ccfnorm = angle
#        ccfs_norm[c] = ccfnorm
#
#    if len(ccfs_norm) > 0:
#        #calculate mean of these, equivalent to the N-day linear stack
#        ccf_mean = np.mean(ccfs_norm, axis=0)
#    
#        #compute amplitude envelope
#        analytic_signal = hilbert(ccf_mean)
#        ampenv = np.abs(analytic_signal)
#        
#        #smooth with hanning window
#        ampenv_smoothed = smooth(ampenv, window_len = fs*smooth_win)
#        #ampenv_resamp = signal.resample(ampenv_smoothed, len(ccf_mean))
#    else:
#        ampenv_smoothed = np.nan
#
#    return ampenv_smoothed

#def compute_AvgPhaseSync(ccfarray, fs, smooth_win = 10, median=True):
#
#    #define taper prior to normalising each ccf by its amplitude envelope
#    taper_perc = 0.1 
#    taper = signal.tukey(len(ccfarray[0]), alpha=taper_perc)
#
#    #array for normalised ccfs
#    ccfs_norm = np.empty(len(ccfarray), dtype=object)
#
#    for c in range(len(ccfs_norm)):
#        
#        #taper pre-hilbert transform
#        ccf_tapered = ccfarray[c] * taper
#        
#        #get phase angle
#        analytic_signal = hilbert(ccf_tapered)
#        angle = np.angle(analytic_signal)
#
#        ccfnorm = angle
#        ccfs_norm[c] = ccfnorm
#
#    if len(ccfs_norm) > 0:
#        
#        #calculate mean of these, equivalent to the N-day linear stack
#        #ccf_mean = np.mean(ccfs_norm, axis=0)
#    
#        nopairs = comb(len(ccfs_norm), 2)
#        diffarray = np.empty(nopairs, dtype=object)
#        paircount = 0
#        for pair in itertools.combinations(ccfs_norm, 2):
#            diffarray[paircount] = 1 - np.sin(np.abs(pair[0]-pair[1])/2)
#            paircount += 1
#
#        if median==True:
#            avgdiff = np.median(diffarray.tolist(), axis=0)        
#        else:
#            avgdiff = np.mean(diffarray.tolist(), axis=0)
#        #std = np.std(diffarray.tolist(), axis=0)
#        ampenv = avgdiff
#
#        #compute amplitude envelope
#        #analytic_signal = hilbert(ccf_mean)
#        #ampenv = np.abs(analytic_signal)
#        
#        #smooth with hanning window
#        ampenv_smoothed = smooth(ampenv, window_len = fs*smooth_win)
#        #ampenv_resamp = signal.resample(ampenv_smoothed, len(ccf_mean))
#    else:
#        ampenv_resamp = np.nan
#
#    return ampenv_smoothed


