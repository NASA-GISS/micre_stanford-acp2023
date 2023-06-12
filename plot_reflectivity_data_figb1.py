#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot_reflectivity_data.py
# Author: McKenna W. Stanford
# Overview: Calculates the 0.01st percentile (effective minimum)
# reflectivity as a function of height and compares it with the
# theoretical minimum assuming irradiance weakens inversely
# proportional to height. Plots Fig. B1 of the manuscript.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#============================================
# Python Imports
#============================================
from give_me_files_and_subfolders import give_me_files_and_subfolders
import numpy as np
from load_sonde_data import load_sonde_data
import datetime
import netCDF4 as nc
import calendar
import sys
import math
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import xarray
import pickle
import seaborn as sns
#============================================
#============================================
basta_path = '/mnt/raid/mwstanfo/micre_data/micre_basta/BASTA_25m/'
#basta_path2 = '/mnt/raid/mwstanfo/micre_basta/BASTA_merged/'
#file = glob.glob(basta_path+'*.nc')[0]
#print(file)
if False:
#if True:

    files = glob.glob(basta_path+'*.nc')
    files2 = glob.glob(basta_path+'*.nc')

    dum = 1
    tt = 1
    nf = len(files)
    for file in files:
        print(tt/nf)
        ncfile = xarray.open_dataset(file)
        dvars = ncfile.data_vars
        ref = ncfile['reflectivity'].values
        flag = ncfile['flag'].values
        flag_coupling = ncfile['flag_coupling'].values
        ncfile.close()
        #ncfile = xarray.open_dataset(files2[tt-1])
        #ref2 = ncfile['reflectivity'].values
        #flag2 = ncfile['flag'].values
        ##ncfile.close()

        tmpid = np.where(flag > 0.)
        #tmpid2 = np.where(flag2 != 0.)
        #ref2[tmpid2] = -999.
        ref[tmpid] = np.nan
        if np.max(flag_coupling) == 1.:
            tmpid = np.where(flag_coupling != 0.)
            ref[:,tmpid] = np.nan
        #tmp = np.append(ref_arr,ref,axis=1)
        if dum == 1:
            height = ncfile['height'].values
            ref_arr = ref
            #ref_arr2 = ref2
            dum = 0
        else:
            ref_arr = np.concatenate((ref_arr,ref),axis=1)
            #ref_arr2 = np.concatenate((ref_arr2,ref2),axis=1)
        tt+=1

    ref_arr[ref_arr == -999.] = np.nan
    #ref_arr2[ref_arr2 == -999.] = np.nan
    
    
    # save these in a pickle dictionary for ease of use later
    # since the processing is expensive
    
    #ref_dict = {'25m':ref_arr,'merged':ref_arr2}
    ref_dict = {'25m':ref_arr,'height':height}
    pkl_file = 'micre_ref_merged_times.p'
    save_path = '/mnt/raid/mwstanfo/'
    pickle.dump(ref_dict, open(save_path+pkl_file,"wb"))
    
    
    
if False:
    save_path = '/mnt/raid/mwstanfo/micre/'
    pkl_file = 'micre_ref_merged_times.p'
    ref_dict = pickle.load(open(save_path+pkl_file,"rb"))

# forgot to save height array to pickle file,
# but it's a quick read so just do it here.
#ncfile = xarray.open_dataset(file)
#height = ncfile['height'].values
#ncfile.close()
height = ref_dict['height']
ref_arr = ref_dict['25m']
#ref_arr_merged = ref_dict['merged']
    
#min_ref = np.nanmin(ref_arr,axis=1)
min_ref = np.nanpercentile(ref_arr,0.01,axis=1)
#min_ref2 = np.nanpercentile(ref_arr_merged,0.01,axis=1)
    
x = 1000.
#Ze_min_at_x_35 = -35.
#Ze_min_at_x_44 = -44.
#Ze_min_at_x_49 = -49.
#Ze_min_at_4km_36 = -36.
Ze_min_at_1km_36 = -36.

z = height
#Ze_min_35 = Ze_min_at_x_35 + 20.*np.log10(z) - 20.*np.log10(x)
##Ze_min_44 = Ze_min_at_x_44 + 20.*np.log10(z) - 20.*np.log10(x)
#Ze_min_49 = Ze_min_at_x_49 + 20.*np.log10(z) - 20.*np.log10(x)
#Ze_min_4km_36 = Ze_min_at_4km_36 + 20.*np.log10(z) - 20.*np.log10(4000.)
Ze_min_1km_36 = Ze_min_at_1km_36 + 20.*np.log10(z) - 20.*np.log10(1000.)
#Ze_min_35[0] = np.nan
#Ze_min_44[0] = np.nan
#Ze_min_49[0] = np.nan
#Ze_min_4km_36[0] = np.nan
Ze_min_1km_36[0] = np.nan
#ref = np.array(ref)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

height_1km,height_1km_id = find_nearest(height,1000.)
min_ref_at_1km = min_ref[height_1km_id]

sns.set_theme()
#sns.set_style('dark')
sns.set_style('ticks')
sns.set(rc={'axes.facecolor':'lavender','axes.edgecolor': 'black','grid.color':'white'})
sns.set_context('talk')  

if True:
    fig = plt.figure(figsize=(8.3,8.3))
    ax1 = fig.add_subplot(111)
    Fontsize=18
    ax1.grid()
    ax1.set_ylabel('Height [km]',fontsize=Fontsize)
    ax1.set_xlabel('Z$_{e,min}$ [dBZ]',fontsize=Fontsize)
    ax1.tick_params(labelsize=Fontsize)
    ax1.plot(min_ref,height/1.e3,lw=3,c='deepskyblue',label='BASTA 0.01$^{st}$ percentile $Z_{e}$\n(1 year during MICRE)')
    #ax1.plot(min_ref2,height/1.e3,lw=4,ls=':',c='cyan',label='Effective Merged Z$_{e,1^{st}percentile}$ (1 year during MICRE)')
    ax1.axhline(1.,lw=4,ls=':',c='blue')
    ax1.text(-67,1.2,'1 km',fontsize=Fontsize*1.5,c='blue')

    ax1.plot(Ze_min_1km_36,height/1.e3,lw=3,c='black',ls='--',label='Theoretical Z$_{e,min}$\n(-36 dBZ @ 1km)')
    ax1.legend(fontsize=Fontsize,loc='center left',bbox_to_anchor=(0,0.5),ncol=1,framealpha=0,facecolor='white')
    ax1.set_title('0.01{} Percentile Z{} @ 1km = {} dBZ'.format('$^{st}$','$_{e}$',str(np.around(min_ref_at_1km,2))),\
                  fontsize=Fontsize*1.2)
    
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.grid(True,which='both',axis='both',c='white')
    plt.show()
    #outfile = 'fig_a1.png'
    outfile = 'fig_b1.png'
    fig_path = '/home/mwstanfo/figures/micre_paper/'
    #plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
    plt.close()
    print('done')

    
    
if False:
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    Fontsize=18
    ax1.grid()
    ax1.set_ylabel('Height [km]',fontsize=Fontsize)
    ax1.set_xlabel('Z$_{e,min}$ [dBZ]',fontsize=Fontsize)
    ax1.tick_params(labelsize=Fontsize)
    ax1.plot(min_ref,height/1.e3,lw=4,c='blue',label='Effective 25-m Z$_{e,1^{st}percentile}$ (1 year during MICRE)')
    #ax1.plot(min_ref2,height/1.e3,lw=4,ls=':',c='cyan',label='Effective Merged Z$_{e,1^{st}percentile}$ (1 year during MICRE)')
    ax1.axhline(1.,lw=4,ls=':',c='k')
    ax1.text(-80,1.2,'1km',fontsize=Fontsize)
    ax1.plot(Ze_min_35,height/1.e3,lw=3,c='red',ls='--',label='Theoretical Z$_{e,min}$ (-35 dBZ @ 1km, SIRTA via Delanoe et al. 2016)')
    ax1.plot(Ze_min_44,height/1.e3,lw=3,c='magenta',ls='--',label='Theoretical Z$_{e,min}$ (-44 dBZ @ 1km, MOBILE via Delanoe et al. 2016)')
    ax1.plot(Ze_min_49,height/1.e3,lw=3,c='darkorange',ls='--',label='Theoretical Z$_{e,min}$ (-49 dBZ @ 1km, BOM via Delanoe et al. 2016)')
    ax1.plot(Ze_min_4km_36,height/1.e3,lw=3,c='green',ls='--',label='Theoretical Z$_{e,min}$ (-36 dBZ @ 4km, via Mace & Protat 2018)')
    #ax1.plot(Ze_min_at_x,1.,'o',markersize=20,c='k')
    #ax1.plot(Ze_min_at_x,1.,'o',markersize=15,c='darkorange')
    #ax1.text(-55,1.4,'Z$_{e,min}$ @ 1 km',fontsize=Fontsize,c='darkorange',fontweight='bold')
    ax1.legend(fontsize=Fontsize-4,loc='lower center',bbox_to_anchor=(0.5,0.5),ncol=1)
    #ax1.set_title('1{} Percentile Z{} @ 1km = {} dBZ'.format('$^{st}$','$_{e}$',str(np.around(min_ref_at_1km,2))),\
    #              fontsize=Fontsize*1.25)

    plt.show()

    
if False:
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    Fontsize=18
    ax1.grid()
    ax2.grid()
    ax1.set_ylabel('Frequency',fontsize=Fontsize)
    ax2.set_ylabel('Count',fontsize=Fontsize)
    ax1.set_xlabel('Reflectivity [dBZ]',fontsize=Fontsize)
    ax2.set_xlabel('Reflectivity [dBZ]',fontsize=Fontsize)
    ax1.tick_params(labelsize=Fontsize)
    ax2.tick_params(labelsize=Fontsize)
    dbz_bins = np.arange(-60,35,5)
    ax1.hist(np.ndarray.flatten(ref_arr),bins=dbz_bins,rwidth=0.85,color='blue',density=True)
    ax2.hist(np.ndarray.flatten(ref_arr),bins=dbz_bins,rwidth=0.85,color='blue',density=False)
    ax1.set_title('PDF',fontsize=Fontsize*1.5)
    ax2.set_title('Histogram',fontsize=Fontsize*1.5)
    plt.show()




