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
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import signal
from scipy.signal import hilbert

def getFilters():
    
    #filtindexes = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']

    filtlowhigh = [[0.01,0.1],[0.03,0.13],[0.05,0.15],[0.1,0.2],[0.15,0.25],[0.2,0.3],[0.25,0.35],[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9],[0.9,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6],[1.6,1.8],[1.8,2.0],[2.0,2.5],[2.5,3.0],[3.0,3.5],[3.5,4.0],[4.0,5.0],[5.0,6.0],[6.0,8.0],[8.0,10.0]]

    centfreqs = np.zeros(len(filtlowhigh))
    for f in range(len(filtlowhigh)):
        cf = np.mean(filtlowhigh[f])
        centfreqs[f] = cf

    return filtlowhigh, centfreqs

def plot_spectogram(inputfile, startdate, enddate, classic=True, demean=False,norm=False, Cor_norm=False, vmax=5000, fig=None, ax=None):

    # df=pd.read_csv('test_df_WIZ.csv',parse_dates=True,index_col=0,header=0)
    df=pd.read_csv(inputfile,parse_dates=True,index_col=0,header=0)
    df=df.resample('1T').mean()

    if norm:
        row_max = df.max(axis=1)
        row_min = df.min(axis=1)
        df = (df -row_min[:, np.newaxis]) / (row_max[:, np.newaxis]-row_min[:, np.newaxis])
        
    if Cor_norm:
        df = (df-df.min())/(df.max()-df.min())
        
    if demean:
        df = df-df.mean(axis=0)
    #fig, ax = plt.subplots(1, 1)

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
    ax.set_title(inputfile)
    ax.set_xlim(pd.Timestamp(startdate), pd.Timestamp(enddate))
    
def plotSNR(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax, axstart=0):
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
    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()

    #set index for getting amplitude each side of ccf
    minlag_amp=5
    posidxamp = np.abs(lagtimes-minlag_amp).argmin()
    negidxamp = np.abs(lagtimes-minlag_amp*-1).argmin()

    #for each day in date range:
    for d in range(len(ccfdates)):
        
        #define booleans for errors that can occur
        errorData = False
        errorFS = False

        day = convertDatetime64ToStr(ccfdates[d])
        
        #for each filter
        snrArray = np.zeros(len(filtlowhigh))
        asymArray = np.zeros(len(filtlowhigh))

        for f in range(len(filtlowhigh)):

            #calculate average SNR between both lag times (positive and negative)
            snr, ampenv, noise, stack, errors = compute_ccf_snr(noisedir, network, stat1, stat2, stacksize, day, filtlowhigh[f], fs, norm=True, loc=loc, component=component)
          
            if errors[0] == True:
                errorData = True
            if errors[1] == True:
                errorFS = True

            #check if array (i.e. not nan)
            if isinstance(snr, (list, tuple, np.ndarray)):

                #get snr values within SNR window (positive and negative)
                snr_p = snr[minidx_psnr:maxidx_psnr+1]
                snr_n = snr[maxidx_nsnr:minidx_nsnr+1]

                #get maximum ampliude on each side of ccf, after the minimum lag time defined
                posamp = np.max(stack[posidxamp+1:])
                negamp = np.max(stack[:negidxamp])

                #average both negative and positive lag time snr
                avgSNR = np.mean([snr_p, snr_n]) 

                #calculate amplitude difference between max of each side of CF
                posmaxsnr = np.max(snr_p)
                negmaxsnr = np.max(snr_n)
                asym_snr = np.log2(posmaxsnr/negmaxsnr)
                asym_amp = np.log2(posamp/negamp)

                #append result for individual filter
                snrArray[f] = avgSNR
                
                if plotAmpSym == True:
                    asymArray[f] = asym_amp
                else:
                    asymArray[f] = asym_snr
            else:
                snrArray[f] = np.nan
                asymArray[f] = np.nan

        if errorData == True:
            print("WARNING: Data missing for stack ending "+day)

        if errorFS == True:
            print("WARNING: YOU HAVE PROVIDED A SAMPLING FREQUENCY DIFFERENT TO THOSE THE CCFs")

        #append result for each day
        snr_freq_array[:,d] = snrArray
        asym_freq_array[:,d] = asymArray

    #snr_freq_array[:,0:stacksize] = np.nan
    #asym_freq_array[:,0:stacksize] = np.nan 

    img1 = ax[axstart].pcolormesh(ccfdates, centfreqs, snr_freq_array[:,:-1], vmin=2, rasterized=True, cmap="jet", shading='auto')
    fig.colorbar(img1, ax=ax[axstart]).set_label('SNR  '+str(minlagwin)+'-'+str(maxlagwin)+' lag')
    ax[axstart].set_yscale('log')
    ax[axstart].set_ylabel('Frequency (Hz)') 

    #get maximum value of asymmetry
    maxasym = np.nanmax(np.abs(asym_freq_array)) 

    if plotAmpSym == True:
        asymLabel = 'ĺog2(pos/neg) Amplitude'
    else:
        asymLabel = 'ĺog2(pos/neg) SNR'

    img2 = ax[axstart+1].pcolormesh(ccfdates, centfreqs, asym_freq_array[:,:-1], rasterized=True, cmap="seismic", shading='auto', vmin=maxasym*-1, vmax=maxasym)
    fig.colorbar(img2, ax=ax[axstart+1]).set_label(asymLabel)
    ax[axstart+1].set_yscale('log')
    ax[axstart+1].set_ylabel('Frequency (Hz)')

def compute_ccf_snr(directory, network, stat1, stat2, stacksize, enddate, frange, fs, filt='01', component='ZZ', smooth_win = 10, loc='00', norm=False):

    #directory containing stacks used in SNR computation
    stackdirpath = directory+'/STACKS/'+filt+'/001_DAYS/'+component+'/'

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

    # print(st)

    if len(st) < stacksize:
       errorData = True

    if len(st) != 0:
                   
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

        #normalize ccfs
        if norm==True:
            #get max for each individual ccf
            ccfs_max = np.max(np.abs(ccfs_1day.tolist()), axis=1)
            ccfs_1day = ccfs_1day / ccfs_max

        #calculate mean of these, equivalent to the N-day stack
        ccf_mean = np.mean(ccfs_1day, axis=0)
        
        #define taper and taper signal for performing hilbert transform
        taper = signal.tukey(len(ccf_mean), alpha=0.2)
        ccf_mean_tapered = ccf_mean * taper

        #compute amplitude envelope
        analytic_signal = hilbert(ccf_mean_tapered)
        ampenv = np.abs(analytic_signal)
        
        #smooth with hanning window
        ampenv_smoothed = smooth(ampenv, window_len = sampfreq*smooth_win)
        ampenv_resamp = signal.resample(ampenv_smoothed, len(ccf_mean))

        #compute noise signal, following Clarke et al. (2011):
        ccfs_1day_squared = np.square(ccfs_1day.tolist())      
        avg_squared = np.mean(ccfs_1day_squared,axis=0) #take avg of squares
        squared_avg = np.square(ccf_mean) #square the average

        noise = np.sqrt((avg_squared-squared_avg)/(len(st)-1))
        
        #smooth noise signal (with same length as earlier smooth)
        noise_smoothed = smooth(noise, window_len = sampfreq*smooth_win)
        noise_resamp = signal.resample(noise_smoothed, len(noise))

        #compute SNR
        snr = ampenv_resamp / noise_resamp
    else:
        snr = np.nan
        ampenv_resamp = np.nan
        noise_resamp = np.nan
        stack = np.nan 

    return snr, ampenv_resamp, noise_resamp, stack, [errorData, errorFS]  

def plotSNRlagtime(directory, network, stat1, stat2, stacksize, startdate, enddate, frange, filt='01', component='ZZ', loc='00', fs=25, maxlag=120):

    #convert dates to UTCDatetime and designate startdate
    enddate_dt = convertDateStrToDatetime(enddate)
    startdate_dt = convertDateStrToDatetime(startdate)
    startdateplot = np.datetime64(startdate)
    enddateplot = np.datetime64(enddate)
        
    #create lag times array
    samprate = 1.0/fs
    x = np.arange(-1*maxlag, maxlag+samprate, samprate)
    minlag=5
    posidx = np.abs(x-minlag).argmin()
    negidx = np.abs(x-minlag*-1).argmin()

    #create date array and reading single day files
    ccfdates = np.arange(startdate_dt, enddate_dt+timedelta(days=1), timedelta(days=1))
    snr_array = np.empty((len(x),len(ccfdates)))
    asym_array_amp = np.zeros(len(ccfdates))
    asym_array_snr = np.zeros(len(ccfdates))
    
    #calculate SNR for each day
    for d in range(len(ccfdates)):
        day = convertDatetime64ToStr(ccfdates[d])
        snr, ampenv, noise, stack = compute_ccf_snr('/home/yatesal/msnoise/'+directory, network, stat1, stat2, stacksize, day, frange, norm=True, loc=loc, component=component)
        snr_array[:,d] = snr
        
        #calculate asymmetry
        posamp = np.max(stack[posidx+1:])
        negamp = np.max(stack[:negidx])

        possnr = np.max(snr[posidx+1:])
        negsnr = np.max(snr[:negidx])

        asym_array_amp[d] = np.log2(posamp/negamp)
        asym_array_snr[d] = np.log2(possnr/negsnr)

    #plot SNR colormap
    fig, ax = plt.subplots(2,1, figsize=(12,7))

    title = network+'_'+stat1+':'+network+'_'+stat2+'('+component+', f'+filt+', m'+str(stacksize)+')'
    fig.suptitle(title)

    #choose max snr for plot
    #vmax = np.nanmax(snr_array)
    #print(vmax)
    #if vmax > 20:
    #    vmax=20
    vmax=20
    img = ax[0].pcolormesh(ccfdates, x, snr_array[:,:-1], rasterized=True, vmin=2, vmax=vmax, cmap="jet", shading='auto')
    #fig.colorbar(img, ax=ax[0])
    fig.colorbar(img, ax=ax).set_label("SNR")
    ax[0].set_ylabel('lag time (s)')
    ax[0].set_xlim(startdateplot,enddateplot)

    ax[1].plot(ccfdates,asym_array_amp, color='black', label='amplitude')
    ax[1].plot(ccfdates,asym_array_snr, color='red', label='snr')
    ax[1].set_xlim(startdateplot,enddateplot)
    ax[1].set_ylabel('log2(pos/neg)')
    ax[1].legend() 

    plt.show()


def example():
    
    #params for which data to use
    startdate = '2018-03-01'
    enddate = '2018-07-01'

    #params for SNR computation
    noisedir = '/home/yatesal/msnoise/kilauea1' #set to directory containing 'STACK' folder of interest
    network = 'HV'
    loc = '--'
    stat1 = 'HAT'
    stat2 = 'OBL' #for single-station, set to same as stat1
    component = 'ZZ'
    stacksize = 10 #not required to match msnoise stacksize
    fs = 50 #set to match CCFs
    maxlag = 120 #set to match CCFs

    minlagwin = 10 #minimum lag for SNR window
    maxlagwin = 40 #maximum lag for SNR window
    plotAmpSym = True #False = asymmetry in CCF SNR, True = asymmetry in CCF amplitude

    #put into array for passing to SNR computation function
    CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]

    #param for covseisnet   
    csndirectory = 'KIL001_HAT_OBL' #set to directory containing covseisnet output
    plotcsn = True

    #param for spectrogram
    plotspectrogram = True
    specdir='/home/yatesal/Scripts/Corentin_RSAM/output/'
    specfname='HAT_HHZ_2018_3_1_2018_7_1.csv' #csv file name
    vmax = 200000 #for plotting
    
    if plotspectrogram and plotcsn:
        numsubplots = 4
        csnax = 1
    elif not plotspectrogram and not plotcsn:
        numsubplots = 2
    else:
        numsubplots = 3
        csnax = 0

    fig, ax = plt.subplots(numsubplots,1, figsize=(10,12))
    
    title = network+'_'+stat1+':'+network+'_'+stat2+'('+component+', m'+str(stacksize)+')'
    fig.suptitle(title)

    print('Plotting SNR data')
    plotSNR(CCFparams, startdate, enddate, minlagwin, maxlagwin, plotAmpSym, fig, ax, axstart=int(numsubplots-2))

    if plotspectrogram == True:
        print('Plotting spectrogram')
        plot_spectogram(specdir+specfname, startdate, enddate, classic=True, demean=False,norm=False, Cor_norm=False, vmax=vmax, fig=fig, ax=ax[0])

    if plotcsn == True:
        print('Plotting spectral width')
        plotSpectralWidth(csndirectory, startdate, enddate, samprate = 25, log=True, count=False, norm=True, fig=fig, ax=ax[csnax])
   
    for i in range(numsubplots): 
       ax[i].set_ylim(0.1,10)

    plt.show()

if __name__ == "__main__":
    main()




