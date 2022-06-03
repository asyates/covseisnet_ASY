import sys
from scipy import signal, interpolate
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform 
from scipy.spatial import distance
from sklearn import preprocessing
from dtaidistance import dtw
from usefulFuncs import *
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm
import csv
from obspy import Stream, Trace

sys.path.insert(1, '/home/yatesal/covseisnet_ASY/')

from covseisnetfunc import *
from SNRanalysis import *

def compute_dmatrix_euclid(stack_n, stack_p):
    
    #calc euclid distance both sides separately (don't need to, but consistent with dtw)
    D_neg = distance.cdist(stack_n, stack_n, 'euclidean')
    D_pos = distance.cdist(stack_p, stack_p, 'euclidean')

    D = np.add(D_neg, D_pos)
    #D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    return D

def compute_dmatrix_manhattan(stack_n, stack_p):

    #calc euclid distance both sides separately (don't need to, but consistent with dtw)
    D_neg = distance.cdist(stack_n, stack_n, 'cityblock')
    D_pos = distance.cdist(stack_p, stack_p, 'cityblock')

    D = np.add(D_neg, D_pos)
    D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    return D

def compute_dmatrix_cc(stack_n, stack_p, sides='both'):
    
    if sides == 'negative':
        data = stack_n
    elif sides == 'positive':
        data = stack_p
    else:   
        #concatenate negative and positive sides (fine for CC)
        data = np.concatenate((stack_n, stack_p), axis=1)

    #compute correlation coefficient matrix    
    corr = np.corrcoef(data)
    corr = (corr + corr.T)/2   # make symmetric
    np.fill_diagonal(corr, 1)  # put 1 on the diagonal

    #convert CC matrix to dissimilarity matrix
    D = 1 - np.abs(corr)
    D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    has_nan = np.isnan(D)
    print(has_nan)
    if has_nan.any() == True:
        print('WARNING: NaN value in distance matrix')

    return D

def compute_dmatrix_dtw(stack_n, stack_p, dtw_win=None):

    D_neg = dtw.distance_matrix_fast(np.array(stack_n,dtype='d'), use_mp=True, psi=None, window=dtw_win)
    D_pos = dtw.distance_matrix_fast(np.array(stack_p,dtype='d'), use_mp=True, psi=None, window=dtw_win)

    #print(D_neg)

    D = np.add(D_neg, D_pos)
    #D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    return D

def compute_dmatrix_ccstretch(ccf_array, CCFparams, minlagwin, maxlagwin, norm=False, max_stretch=0.01, step=0.0025): 

    # empty array that will store max values of CC
    maxCC = np.zeros((len(ccf_array), len(ccf_array))) 
    dvv_array = np.zeros((len(ccf_array), len(ccf_array)))

    #create list of all stretch values to apply
    stretch_values = np.arange(-max_stretch, max_stretch+step, step)
   
    fs = CCFparams[7]
    maxlag = CCFparams[8]

    stack_n, stack_p = sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=norm)
    #concatenate negative and positive sides (fine for CC)
    data = np.concatenate((stack_n, stack_p), axis=1)


    for i,ccf in enumerate(ccf_array):
        print(i) 
        CC_temp = np.zeros(len(ccf_array))
        dvv_temp = np.zeros(len(ccf_array))
        for value in stretch_values:
            
            ccf_stretched = stretchccf_stretchtime(ccf,value, maxlag, fs)
            stack_n, stack_p = sliceCCFs(CCFparams, np.array([ccf_stretched]), minlagwin, maxlagwin, norm=norm)
            data0 = np.concatenate((stack_n, stack_p), axis=1)

            CC = 1 - distance.cdist(data0,np.stack(data), 'correlation')
            
            CC_temp = np.maximum(CC_temp, CC) #maybe for this, should only be looking at positive CC values

            
            dvv_temp = update_best_dvv(CC, CC_temp, value, dvv_temp)
            #print(dvv_temp)

                
        maxCC[i,:] = CC_temp
        maxCC[:,i] = CC_temp


        dvv_array[i,:] = dvv_temp
        dvv_array[i,:] = dvv_temp

        
    
    ##convert CC matrix to dissimilarity matrix
    np.fill_diagonal(maxCC, 1)  # put 1 on the diagonal
    D = 1 - np.abs(maxCC)

    return D

def update_best_dvv(corr, maxCC, stretch, dvv_array):

    for i in range(len(corr[0])):
        #print(i, corr[0][i], maxCC[0][i])
        #print(corr[0][i], maxCC[0][i])
        if corr[0][i] == maxCC[0][i]:
            dvv_array[i] = stretch
            #print(i)
    return dvv_array

def stretchdata(ccf_array, value, fs, maxlag):

    ccf_array_stretched = np.empty(len(ccf_array), dtype=object)

    for i, ccf in enumerate(ccf_array):
        ccf_stretched = stretchccf_stretchtime(ccf,value, maxlag, fs)
        ccf_array_stretched[i] = ccf_stretched

    return ccf_array_stretched

def stretchccf_stretchtime(ccf, dvv , maxlag, fs):

    scalefact = 1+(1*dvv)

    #maxlag = 120
    samprate = 1.0/fs

    lagtimes_orig = np.arange(-1*maxlag, maxlag+samprate, samprate)
    lagtimes_new = np.arange(-1*maxlag, maxlag+samprate, samprate)*scalefact

    
    #print(lagtimes_new)
    #print(len(lagtimes_new), len(ccf))

    f = interpolate.interp1d(lagtimes_new, ccf, fill_value='extrapolate')

    #fig0, ax0 = plt.subplots()

    #ax0.plot(lagtimes_orig, ccf)
    #ax0.plot(lagtimes_orig, f(lagtimes_orig))

    #plt.show()

    return f(lagtimes_orig)


def sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=False):

    #create lag time array using sampling rate and maxlag
    samprate = 1.0/CCFparams[7]
    maxlag = CCFparams[8]
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)

    #get minimum and maximum index for snr windows, from minlagwin and maxlagwin
    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()

    #slice arrays to be just times of interest, both positive and negative side of CCF
    ccf_array = np.vstack(np.array(ccf_array))
      
    if norm == True: #normalise using maximum value
        max_values = np.max(np.abs(ccf_array),axis=1) #gets max value in each row/CCF
        ccf_array = ccf_array / max_values[:,None]

    stack_p = ccf_array[:,minidx_psnr:maxidx_psnr+1]
    stack_n = ccf_array[:,maxidx_nsnr:minidx_nsnr+1]
 
    return stack_n, stack_p 

def getCCFData(CCFparams, frange, startdate, enddate, step=1, filt='01', phase=False, filtdata=False):

    #convert string dates to datetime64
    startdatedt = np.datetime64(startdate)
    enddatedt = np.datetime64(enddate)

    #create array of datetimes between start and enddate, spaced by defined step
    days = np.arange(startdatedt, enddatedt, np.timedelta64(step, 'D'))
    
    ccf_array = []
    days_used = []

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

    #for each day, get stack corresponding to defined CCFparams, story in ccf_array
    for i, day in enumerate(days):

        day = convertDatetime64ToStr(day) #convert back to string for getCCFStack function call
       
        if frange != [0.0, 0.0] and filtdata==True: 
            stack, singles_array = getCCFStack(noisedir, network, stat1, stat2, stacksize, day, frange, fs, filt=filt, component=component, loc=loc, stacksuffix='')
        else:
 
            #format filter number for filepath (convert to string, pad 0 if neccessary)
            if int(filt) < 10:
                filt_fmt = '0'+str(int(filt))
            else:
                filt_fmt = str(int(filt))

            #format stacksize for filepath
            if stacksize < 10:
                stacksize_fmt = '00'+str(stacksize)
            elif stacksize < 100:
                stacksize_fmt = '0'+str(stacksize)
            else:
                stacksize_fmt = str(stacksize)

            stacksuffix=''
            stackdirpath = noisedir+'/STACKS'+stacksuffix+'/'+filt_fmt+'/'+stacksize_fmt+'_DAYS/'+component+'/'
            
            statdir = network+'.'+stat1+'.'+loc+'_'+network+'.'+stat2+'.'+loc+'/'
            fname = day+'.MSEED'
            fullpath = stackdirpath+statdir+fname
            
            st = Stream()

            if os.path.isfile(fullpath):
                st += read(fullpath)
                stack = np.array(st[0].data)
            else:
                print(fullpath + ' missing')
                stack = np.nan

        if phase == True:
            stack = convertToPhase(stack)

        #check if array (i.e. not nan)
        if isinstance(stack, (list, tuple, np.ndarray)):
            ccf_array.append(stack)
            days_used.append(days[i])

    return ccf_array, days_used


def clusterCCFs(ccf_array, days, CCFparams, frange, startdate, enddate, minlagwin, maxlagwin, cluster_method='cc', step=1, sides='both', norm=False, thresholds=[0.3,0.5,0.7], fs=25, dvvchange_array = np.array([]), fig=None, axes=[], writeout=False, filename=''):
    
    if cluster_method == 'euclid':
        
        stack_n, stack_p = sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=norm)
        D = compute_dmatrix_euclid(stack_n, stack_p) 

    elif cluster_method == 'manhattan':
        
        stack_n, stack_p = sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=norm)
        D = compute_dmatrix_manhattan(stack_n, stack_p) 

    elif cluster_method == 'cc':

        stack_n, stack_p = sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=norm)
        D = compute_dmatrix_cc(stack_n, stack_p, sides=sides) 

    elif cluster_method == 'ccstretch':
        D = compute_dmatrix_ccstretch(ccf_array, CCFparams, minlagwin, maxlagwin, norm=norm)

    elif cluster_method == 'ccspectra':
        D = compute_dmatrix_ccspectra(ccf_array, CCFparams, minlagwin, maxlagwin, frange)

    elif cluster_method == 'dtw':

        #compute window max for dtw
        maxdvv_win = 0.05
        samprate = 1.0/fs
        dtw_win = int((maxlagwin*maxdvv_win) / samprate)

        stack_n, stack_p = sliceCCFs(CCFparams, ccf_array, minlagwin, maxlagwin, norm=norm)
        #print(stack_n[0])
        D = compute_dmatrix_dtw(stack_n, stack_p, dtw_win=dtw_win)
    else:
        print('''Unrecognized clustering method, choose one of 'euclid', 'cc','ccstretch','dtw'.''') 
        return

    
    if writeout == True:
        dfilename=filename+'_dmatrix'
        np.save('dmatrix_outputs/'+dfilename, D)    

    fig0, ax0 = plt.subplots()
    cmap = cm.get_cmap('viridis_r',12)
    print(D)
    psm = ax0.pcolormesh(days, days, D, cmap=cmap, rasterized=True, shading='auto')
    plt.gca().invert_yaxis()
    fig.colorbar(psm, ax=ax0)
    

    #perform linkage agglomorative clustering
    linkagemethod="average"
    H = hierarchy.linkage(squareform(D), method=linkagemethod)
        
    labels = np.empty(len(thresholds), dtype=object) #array for cluster labels
    
    #cluster using the defined thresholds   
    for i,thres in enumerate(thresholds):
        labels[i] = hierarchy.fcluster(H, thres, criterion='distance')

    if writeout==True:
        for i, thres in enumerate(thresholds):
            filepath='cluster_outputs/'
            writeout_cluster(filepath, filename+'_'+str(thres), days, labels[i], thres, cluster_method)

    #create color scales for each threshold
    color_ranges = np.empty(len(labels), dtype=object)
    for l in range(len(labels)):
        
        num_fam = np.max(labels[l])
        #cmap = ListedColormap(sns.color_palette('colorblind').as_hex())
        cmap = plt.get_cmap('rainbow')
        color_ranges[l] = cmap(np.linspace(0.0, 1.0, int(num_fam)))

    #plot dendrogram
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    print(len(days))
    #print(color_ranges[0].tolist())
    hierarchy.set_link_color_palette([cm.colors.rgb2hex(rgb) for rgb in color_ranges[0]])
    #dn = hierarchy.dendrogram(H, color_threshold=thresholds[0], above_threshold_color='black', distance_sort=False, count_sort=True, no_labels=True)

    dn = hierarchy.dendrogram(H, color_threshold=thresholds[0], above_threshold_color='black', no_labels=True)

    thres_plot = np.arange(0.01,1+0.01,0.01)
    num_groups = np.zeros(len(thres_plot))
    for i, thres in enumerate(thres_plot):
        ltemp = hierarchy.fcluster(H, thres, criterion='distance')
        num_groups[i] = np.max(ltemp)    


    fig1, ax1 = plt.subplots()
    ax1.plot(thres_plot, num_groups)
    #ax1.scatter(thres_plot, num_groups)
    ax1.set_xlabel('distance threshold')
    ax1.set_ylabel('no. clusters')
    ax1.set_yscale('log')
    for thres in thresholds:
        ax1.axvline(x = thres, color = 'red', linewidth=2, linestyle = 'dashed')

    #define plot for showing clusters
    if dvvchange_array.size == 0 :
        #fig, ax = plt.subplots(len(thresholds)+1, 1, constrained_layout=True)
        #plotInterferogram(CCFparams, startdate, enddate, frange, fig=fig, ax=ax[0])
        print('no longer doing anything')
    else:
        #fig, ax = plt.subplots(len(thresholds)+2, 1, constrained_layout=True)
        #ax[len(thresholds)+1].plot(days, -100*dvvchange_array)
        #ax[len(thresholds)+1].set_ylabel('dv/v (%)')
        #ax[len(thresholds)+1].axvspan(days[160], days[180], color='red',alpha=0.5) #HARD CODED

        #create lag time array using sampling rate and maxlag
        #samprate = 1.0/CCFparams[7]
        #maxlag = CCFparams[8]
        #lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
        
        #plot_interferogram_syn(lagtimes, ccf_array, days, fig=fig, ax=ax[0], maxlag=60)
        print('no longer doing anything')


    #plot interferogram of CCFs corresponding to defined CCFparams
    print(startdate, enddate)
    
    #print(num_fam)
    #print(color_ranges)
    #print(len(labels))

    #color_ranges = np.array([np.array(['red','green','blue'])])
    #print(color_ranges[0][1])
    
    for d in range(len(days)):     
        if d == 0:
            continue
        else:
            for l, label in enumerate(labels): 
                #axes[l].axvspan(days[d-1], days[d], color=getLabelColour(labels[l][d]-1), alpha=0.5)
                #print(d, color_ranges[l][label[d]-1])
                axes[l].axvspan(days[d-1], days[d], color=color_ranges[l][label[d]-1], alpha=0.7, lw=0)


    for l in range(len(labels)):
        axes[l].set_title('Distance threshold = '+str(thresholds[l]))
        axes[l].scatter(days, labels[l], color='black')
        axes[l].set_ylabel('Cluster No.')
    
    #draw lines for min and max lag time of coda in CCFs
    coda_idx = 0
    #ax[coda_idx].axhline(y = minlagwin, color = 'black', linewidth=2, linestyle = 'dashed')
    #ax[coda_idx].axhline(y = maxlagwin, color = 'black', linewidth=2, linestyle = 'dashed')
    #ax[coda_idx].axhline(y = minlagwin*-1, color = 'black', linewidth=2, linestyle = 'dashed')
    #ax[coda_idx].axhline(y = maxlagwin*-1, color = 'black', linewidth=2, linestyle = 'dashed')
    #ax[coda_idx].set_ylim(-1.5*maxlagwin, 1.5*maxlagwin)

    #set title
    stat1_full = CCFparams[1]+'.'+CCFparams[3]+'.'+CCFparams[2]
    stat2_full = CCFparams[1]+'.'+CCFparams[4]+'.'+CCFparams[2]

    #ax[coda_idx].set_title(stat1_full+'_'+stat2_full+' comp'+CCFparams[5]+' M'+str(CCFparams[6])+' '+str(frange))

    for i in range(len(axes)):
        axes[i].set_xlim(np.datetime64(days[0]), np.datetime64(days[-1]))
        axes[i].yaxis.get_major_locator().set_params(integer=True)

    #plt.show()

def getLabelColour(num):
    
    #function returns a color corresponding to the index provided as input.

    colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','thistle','cyan','lime','aqua','magenta','navy','yellow','palegreen','silver','gold','teal','lightsteelblue', 'firebrick','wheat','rosybrown','plum','dodgerblue','chartreuse','turquoise','cadetblue','linen','rebeccapurple','azure','papayawhip']

    num_colors = len(colors)
    idx = num % num_colors

    return colors[idx]

def plot_interferogram_syn(lagtimes, ccfs, days, fig=None, ax=None, ax_cb=None, maxlag=120):

    if ax_cb == None:
        ax_cb = ax

    df = pd.DataFrame(np.array(ccfs).real.tolist(), index=days, columns=lagtimes)
    df = df.dropna()

    #define the 99% percentile of data for visualisation purposes
    clim = df.mean(axis='index').quantile(0.99)

    print(fig)
    print(ax)
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(12,8))
        plot = True
    else:
        plot = False

    img = ax.pcolormesh(df.index, df.columns, df.values.T, vmin=-clim, vmax=clim, rasterized=True, cmap='seismic')
    #fig.colorbar(img, cax=ax_cb).set_label('')
  
    print('ccf for interferogram') 
    print(ccfs[0])

    #plt.colorbar()
    #ax.set_title('Interferogram')
    ax.set_ylabel('Lag Time (s)')
    ax.set_ylim(maxlag*-1, maxlag)

    if plot == True:
        plt.show()

def writeout_cluster(filepath, filename, days, labels, threshold, method):

    with open(filepath+filename+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['day','cluster','threshold','method'])
        for i, day in enumerate(days):
            writer.writerow([day, labels[i], threshold, method]) 

def example(statpair,method, frange, thresholds=[0.3,0.5,0.7], filt='01', filtdata=False, phase=False, norm=False, lagwin=[10,45]):

    noisedir = '/home/yatesal/msnoise/piton04' #set to directory containing 'STACK' folder of interest
    network = 'YA'
    loc = '00'
    filt = filt
    stat1 = statpair[0]
    stat2 = statpair[1] #for single-station, set to same as stat1
    component = 'ZZ'
    stacksize = 10 #not required to match msnoise stacksize
    fs = 25 #set to match CCFs
    maxlag = 120 #set to match CCFs
    
    #put into array for passing to SNR computation function
    CCFparams = [noisedir, network, loc, stat1, stat2, component, stacksize, fs, maxlag]

    minlagwin = lagwin[0] #minimum lag for SNR window
    maxlagwin = lagwin[1] #maximum lag for SNR window

    startdate='2009-10-01'
    enddate='2011-05-01'
    step=3

    ccf_array, days = getCCFData(CCFparams, frange, startdate, enddate, step=step, filt=filt, phase=phase, filtdata=filtdata)
        

    fig, ax = plt.subplots(len(thresholds)+2, 1, constrained_layout=True)

    #csndirectory='PIT017'
    #plotSpectralWidth(csndirectory, startdate, enddate, log=True, count=False, norm=True, fig=fig, ax=ax[0])

    #plotInterferogram(CCFparams, startdate, enddate, frange, fig=fig, ax=ax[1])

    #create lag time array using sampling rate and maxlag
    samprate = 1.0/CCFparams[7]
    maxlag = CCFparams[8]
    lagtimes = np.arange(-1*maxlag, maxlag+samprate, samprate)
    
    plot_interferogram_syn(lagtimes, ccf_array, days, fig=fig, ax=ax[1], maxlag=60)

    workdir = '/home/yatesal/msnoise/piton04/'
    #plot_dvv_stretch(workdir,'YA', 'UV05', 'UV12', 10, startdate, enddate, filt=2, ax=ax[0], fig=fig)
    plot_dvv_stretch(workdir,'YA', stat1, stat2, 10, startdate, enddate, filt=int(filt), ax=ax[0], fig=fig) 
    ax[0].set_xlim(np.datetime64(days[0]), np.datetime64(days[-1]))

    plotPitonEruptions(ax[0])

    
    coda_idx = 1
    ax[coda_idx].axhline(y = minlagwin, color = 'black', linewidth=2, linestyle = 'dashed')
    ax[coda_idx].axhline(y = maxlagwin, color = 'black', linewidth=2, linestyle = 'dashed')
    ax[coda_idx].axhline(y = minlagwin*-1, color = 'black', linewidth=2, linestyle = 'dashed')
    ax[coda_idx].axhline(y = maxlagwin*-1, color = 'black', linewidth=2, linestyle = 'dashed')
    ax[coda_idx].set_ylim(-1.5*maxlagwin, 1.5*maxlagwin)

    #set title
    stat1_full = CCFparams[1]+'.'+CCFparams[3]+'.'+CCFparams[2]
    stat2_full = CCFparams[1]+'.'+CCFparams[4]+'.'+CCFparams[2]


    clusterCCFs(ccf_array, days, CCFparams, frange, startdate, enddate, minlagwin, maxlagwin, cluster_method=method, step=step, norm=norm, thresholds=thresholds, fig=fig, axes=[ax[2],ax[3]])    

    plt.show()

