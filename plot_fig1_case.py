#=========================================================
# plot_fig1_case.py
# Merges radar, ceilometer from ARM and from U. of Canterbury,
# soundings, surface meteorological variables, and
# satellite data. Plots specific case, which 
# Author: McKenna W. Stanford
#=========================================================

#--------------------------------
# Imports
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray
import datetime
import calendar
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib
import pickle
import pandas as pd
import os
from file_struct import file_struct as fs
from load_sonde_data import load_sonde_data
from give_me_files_and_subfolders import give_me_files_and_subfolders
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator as nn
from scipy.interpolate import griddata as griddata
from calculate_theta_and_more import calculate_theta_and_more
import pandas
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import palettable

#--------------------------------------------
#--------------------------------------------

#--------------------------------------------
# Functions
#--------------------------------------------
def toTimestamp(d):
    return calendar.timegm(d.timetuple())

def toDatetime(d):
    return datetime.datetime.utcfromtimestamp(d)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx   

# function to make serial date numbers which are the number of days that have passed
# since epoch beginning given as days.fraction_of_day
def datenum(d):
        return 366 + d.toordinal() + (d - datetime.datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)
#--------------------------------------------
#--------------------------------------------


#--------------------------------------------
# Grab BASTA files
#--------------------------------------------
basta_path = '/mnt/raid/mwstanfo/micre_data/micre_basta/BASTA_25m/'
basta_files = glob.glob(basta_path+'*.nc')
basta_files = sorted(basta_files)
basta_files = np.array(basta_files)

basta_dates_dt = []
for ii in range(len(basta_files)):
    fname = basta_files[ii]
    tmp_str = fname.split('_')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('.')
    tmp_str = tmp_str[0]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:8])
    basta_dates_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day,0,0,0))
basta_dates_dt = np.array(basta_dates_dt)
#basta_datenum = [datenum(basta_dates_dt[dd]) for dd in range(len(basta_dates_dt))]
#basta_datenum = np.array(basta_datenum)  
print(len(basta_dates_dt))
#--------------------------------------------
# End obtention of BASTA files
#--------------------------------------------

#--------------------------------------------
# Grab ARM ceilometer files
#--------------------------------------------
ceil_path = '/mnt/raid/mwstanfo/micre_data/micre_ceil/'
ceil_files = glob.glob(ceil_path+'*.nc')
ceil_files = sorted(ceil_files)
ceil_files = np.array(ceil_files)

ceil_dates_dt = []
for ii in range(len(ceil_files)):
    fname = ceil_files[ii]
    tmp_str = fname.split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('.')
    tmp_str = tmp_str[2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:8])
    ceil_dates_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day,0,0,0))
ceil_dates_dt = np.array(ceil_dates_dt)

# Limit ceilometer files to encompass only BASTA dates
tmpid = np.where((ceil_dates_dt >= basta_dates_dt[0]) & (ceil_dates_dt <= basta_dates_dt[-1]))[0]
ceil_files = ceil_files[tmpid]
ceil_dates_dt = ceil_dates_dt[tmpid]

#--------------------------------------------
# End obtention of ARM ceilometer files
#--------------------------------------------

#--------------------------------------------
# Grab AAD ceilometer files
#--------------------------------------------
aad_ceil_path = '/mnt/raid/mwstanfo/micre_data/aad_ceil/AAS_4292_Macquarie_Ceilometer/'
aad_ceil_files = glob.glob(aad_ceil_path+'*.nc')
aad_ceil_files = sorted(aad_ceil_files)
aad_ceil_files = np.array(aad_ceil_files)

aad_ceil_dates_dt = []
aad_ceil_times_dt = []
for ii in range(len(aad_ceil_files)):
    fname = aad_ceil_files[ii]
    tmp_str = fname.split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('.')
    tmp_str = tmp_str[0]
    tmp_str = tmp_str.split('-')
    tmp_year = int(tmp_str[0])
    tmp_month = int(tmp_str[1])
    tmp_dum = tmp_str[2]
    tmp_dum = tmp_dum.split('T')
    tmp_day = int(tmp_dum[0])
    tmp_hour_min_str = tmp_dum[1][0:4]
    tmp_hour = int(tmp_hour_min_str[0:2])
    tmp_min = int(tmp_hour_min_str[2:4])
    aad_ceil_times_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_min))
    aad_ceil_dates_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day))
    
aad_ceil_times_dt = np.array(aad_ceil_times_dt)
aad_ceil_dates_dt = np.array(aad_ceil_dates_dt)

# Limit ceilometer files to encompass only BASTA dates
tmpid = np.where((aad_ceil_dates_dt >= basta_dates_dt[0]) & (aad_ceil_dates_dt <= basta_dates_dt[-1]))[0]
aad_ceil_times_dt = aad_ceil_times_dt[tmpid] 
aad_ceil_dates_dt = aad_ceil_dates_dt[tmpid] 
aad_ceil_files = aad_ceil_files[tmpid]

#--------------------------------------------
# End obtention of AAD ceilometer files
#--------------------------------------------


#--------------------------------------------
# Grab surface meteorology files
#--------------------------------------------
sfc_path = '/mnt/raid/mwstanfo/micre_data/micre_sfc/'
sfc_files = glob.glob(sfc_path+'*.nc')
sfc_files = sorted(sfc_files)
sfc_files = np.array(sfc_files)

sfc_dates_dt = []
for ii in range(len(sfc_files)):
    fname = sfc_files[ii]
    tmp_str = fname.split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('.')
    tmp_str = tmp_str[2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:8])
    sfc_dates_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day,0,0,0))
sfc_dates_dt = np.array(sfc_dates_dt)

# Limit sfc met files to encompass only BASTA dates
tmpid = np.where((sfc_dates_dt >= basta_dates_dt[0]) & (sfc_dates_dt <= basta_dates_dt[-1]))[0]
sfc_dates_dt = sfc_dates_dt[tmpid] 
sfc_files = sfc_files[tmpid]
#--------------------------------------------
#-------------------------------------------- 


#--------------------------------------------
# Grab satellite files
#--------------------------------------------
sat_path = '/mnt/raid/mwstanfo/micre_data/visst_gridded/'
sat_files = glob.glob(sat_path+'*.nc')
sat_files = sorted(sat_files)
sat_files = np.array(sat_files)

sat_dates_dt = []
for ii in range(len(sat_files)):
    fname = sat_files[ii]
    tmp_str = fname.split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('.')
    tmp_str = tmp_str[0]
    tmp_str = tmp_str.split('_')
    tmp_str = tmp_str[2:]
    tmp_year = int(tmp_str[0])
    tmp_month = int(tmp_str[1])
    tmp_day = int(tmp_str[2])
    sat_dates_dt.append(datetime.datetime(tmp_year,tmp_month,tmp_day,0,0,0))
sat_dates_dt = np.array(sat_dates_dt)

# sort files according to dates
sort_id = np.argsort(sat_dates_dt)
sat_dates_dt = sat_dates_dt[sort_id]
sat_files = sat_files[sort_id]

# Limit sat files to encompass only BASTA dates
tmpid = np.where((sat_dates_dt >= basta_dates_dt[0]) & (sat_dates_dt <= basta_dates_dt[-1]))[0]
sat_dates_dt = sat_dates_dt[tmpid] 
sat_files = sat_files[tmpid]
#--------------------------------------------


#--------------------------------------------
# Grab sounding files
#-------------------------------------------- 
path = '/mnt/raid/mwstanfo/micre_data/micre_soundings/'
sonde_files = glob.glob(path+'*.nc')
sonde_files = sorted(sonde_files)
sonde_files = np.array(sonde_files)
nf = len(sonde_files)

sonde_dates_dt = []
sonde_times_dt = []
for ff in range(nf):
    sonde_file = sonde_files[ff]
    sonde_file = str.split(sonde_file,'/')[-1]
    sonde_file = str.split(sonde_file,'.')[0]
    sonde_file_str1 = str.split(sonde_file,'_')[0]
    sonde_file_str2 = str.split(sonde_file,'_')[1]
    year = int(sonde_file_str1[0:4])
    month = int(sonde_file_str1[4:6])
    day = int(sonde_file_str1[6:8])
    hour = int(sonde_file_str2[0:2])
    minute = int(sonde_file_str2[2:4])
    sonde_time_dt = datetime.datetime(year,month,day,hour,minute)
    sonde_date_dt = datetime.datetime(year,month,day)
    sonde_dates_dt.append(sonde_date_dt)
    sonde_times_dt.append(sonde_time_dt)
sonde_dates_dt = np.array(sonde_dates_dt)
sonde_times_dt = np.array(sonde_times_dt)

# Limit sonde files to be
#--------------------------------------------
#-------------------------------------------- 


#--------------------------------------------
# Read in cluster information
#-------------------------------------------- 
if False:
    cluster_file ='/home/mwstanfo/cluster_information_all.xlsx'
    clusters = pandas.read_excel(cluster_file)
    clusters = np.array(clusters)
    cluster_number = clusters[:,0]
    cluster_dates = clusters[:,1]
    cluster_id = clusters[:,2]

    cluster_dates_dt = []
    cluster_times_dt = []
    for cluster_date in cluster_dates:
        tmpstr1,tmpstr2 = str.split(cluster_date,'_')
        tmpyear = int(tmpstr1[0:4])
        tmpmonth = int(tmpstr1[4:6])
        tmpday = int(tmpstr1[6:8])
        tmphour = int(tmpstr2[0:2])
        tmpminute = int(tmpstr2[2:4])
        cluster_times_dt.append(datetime.datetime(tmpyear,tmpmonth,tmpday,tmphour,tmpminute))
        cluster_dates_dt.append(datetime.datetime(tmpyear,tmpmonth,tmpday))
    cluster_dates_dt = np.array(cluster_dates_dt)
    cluster_times_dt = np.array(cluster_times_dt)



#--------------------------------------------------------------
# Use radar to define timeline
# and exclude days without any soundings
#--------------------------------------------------------------
#dates = basta_dates_dt.copy()
dumid = np.where((sonde_dates_dt >= basta_dates_dt[0]) & (sonde_dates_dt <= basta_dates_dt[-1]))
sonde_dates_dt = sonde_dates_dt[dumid]
sonde_times_dt = sonde_times_dt[dumid]
sonde_files = sonde_files[dumid]
dates = sonde_dates_dt.copy()
times = sonde_times_dt.copy()
unique_dates = np.unique(dates)

# Discard dates that don't have any soundings at all.
# also limit basta_dates and files to the new dates
if False:
    new_dates = []
    dum = 0
    mask_dates = np.zeros_like(basta_dates_dt)
    for date in dates:
        sonde_id = np.where(sonde_dates_dt == date)[0]
        if len(sonde_id) > 0:
            new_dates.append(date)
            mask_dates[dum] = 1
        dum+=1
    new_dates = np.array(new_dates)

    dates = new_dates
    tmpid = np.where(mask_dates == 1)[0]
    basta_dates_dt = basta_dates_dt[tmpid]
    basta_files = basta_files[tmpid]
    del new_dates


#=============================================================
#=============================================================
#=============================================================
#---------------------------------
# MASTER LOOP
#---------------------------------
# Loop through Sounding files. This will control everything else.
#
#=============================================================
#=============================================================
#=============================================================
#unique_dates = unique_dates[24:]
#unique_dates = unique_dates[60:]

target_date = datetime.datetime(2016,11,6)
dumid = np.where(unique_dates == target_date)[0][0]
unique_dates = unique_dates[dumid:]
y_height_const = 1.5
ii = 0
for date in unique_dates:
    
    print(ii)

        
    print('Sonde date:',date)

    
    #===========================================
    # Begin sounding block
    #===========================================
    print('Begin sounding processing.')
    

    #------------------------------------------    
    # First find sounding dates that match the
    # current BASTA date. Due to filtering above
    # of BASTA dates where soundings don't exist
    # at all, there should always be at least
    # one sounding on a given day.
    #------------------------------------------    
    sonde_id = np.where(sonde_dates_dt == date)[0]
        
    # get times of soundinges and files
    current_date_sonde_times_dt = sonde_times_dt[sonde_id]
    current_date_sonde_files = sonde_files[sonde_id]
    num_current_soundings = len(current_date_sonde_times_dt)

    sonde_temperature = []
    sonde_pressure = []
    sonde_rh = []
    sonde_u = []
    sonde_v = []   
    sonde_wind_dir = []
    sonde_wind_speed = []
    sonde_height = []
    sonde_q = []
    sonde_theta = []
    sonde_theta_e = []
    sonde_rh_i = []
    sonde_max_alt = []
    sonde_time_dt = []
    sonde_l_v = []
    sonde_w_s = []
    sonde_e = []
    sonde_time_dt_long = []
    

    jj_ind = []  
    for jj in range(num_current_soundings):
        
        current_date_sonde_file = current_date_sonde_files[jj]
        current_date_sonde_file = current_date_sonde_file.split('/')[-1]
        path = '/mnt/raid/mwstanfo/micre_data/micre_soundings/'

        file_size = os.stat(path+current_date_sonde_file).st_size/1.e3
        fstruct = fs(current_date_sonde_file,path,file_size)
        Sondetmp = load_sonde_data(fstruct)
        
        max_alt = np.max(Sondetmp['alt'])
        if max_alt < 10.:
            print('Sonde failed to reach 10 km. Therefore omitting this sounding.')
            jj_ind.append(jj)
            continue
        
        sonde_temperature.append(Sondetmp['drybulb_temp'])
        sonde_pressure.append(Sondetmp['pressure'])
        sonde_rh.append(Sondetmp['RH'])
        sonde_u.append(Sondetmp['u_wind'])
        sonde_v.append(Sondetmp['v_wind'])
        sonde_wind_dir.append(Sondetmp['wind_direction'])
        sonde_wind_speed.append(Sondetmp['wind_speed'])
        sonde_height.append(Sondetmp['alt'])
        Moretmp = calculate_theta_and_more(Sondetmp['drybulb_temp'],Sondetmp['pressure'],\
                                           RH=Sondetmp['RH'],use_T_K=True,\
                                          sat_pres_formula='Emmanuel')
        sonde_q.append(Moretmp['q'])
        sonde_theta.append(Moretmp['Theta'])
        sonde_theta_e.append(Moretmp['Theta_e'])
        sonde_rh_i.append(Moretmp['RH_i'])
        sonde_l_v.append(Moretmp['L_v'])
        sonde_w_s.append(Moretmp['w_s'])
        sonde_e.append(Moretmp['e'])
        sonde_time_dt_long.append(Sondetmp['time'])
        sonde_time_dt.append(current_date_sonde_times_dt[jj])

        sonde_max_alt.append(max_alt)
        
    if np.size(jj_ind) > 0.:
        current_date_sonde_files = np.delete(current_date_sonde_files,jj_ind)
        current_date_sonde_times_dt = np.delete(current_date_sonde_times_dt,jj_ind)
        sonde_id = np.delete(sonde_id,jj_ind)
        num_current_soundings = len(current_date_sonde_files)
        
    # in the case that all of the current soundings were ommitted due to errors,
    # then we'll skip this date. Do this by re-checking num_current_soundings.
    if num_current_soundings == 0:
        ii+=1
        print('Further analysis omits this date due to bad sonde data.')
        print('')
        continue
    else:
        pass

    #===========================================
    # End sounding block
    #===========================================
    print('Completed sounding processing.')
    
    #======================================================
    # Begin radar block
    #======================================================
    print('Begin radar processing.')
    dumid = np.where(basta_dates_dt == date)
    if np.size(dumid) == 0.:
        print('No radar data on this date. Ommitting sounding.')
        ii+=1
        continue    
    dumid = dumid[0][0]

    ncfile = xarray.open_dataset(basta_files[dumid],decode_times=False)
    basta_time_dims = ncfile.dims['time'] # will be variable according to up-time
    basta_height_dims = ncfile.dims['height'] # should always be 480
    basta_ref = np.array(ncfile['reflectivity'].copy())
    basta_vel = np.array(ncfile['velocity'].copy())
    basta_flag = np.array(ncfile['flag'].copy())
    basta_flag_coupling = np.array(ncfile['flag_coupling'].copy()) # 0: no coupling (good); 1: coupling (bad)
    basta_noise_level = np.array(ncfile['noise_level'].copy()) # 0: good data; 1-9: bad data; -1: no data
    basta_time_sec_since_00Z = np.array(ncfile['time'].copy())
    basta_height = np.array(ncfile['height'].copy()) # 25-m resolution beginning at 12.5 m (mid-bin)
    ### ends at 11987.5 m, so 12 km
    ncfile.close()    


    tmp_basta_time_ts = toTimestamp(datetime.datetime(date.year,\
                                       date.month,\
                                       date.day))
    tmp_basta_time_ts = tmp_basta_time_ts + basta_time_sec_since_00Z
    basta_time_dt = [toDatetime(tmp_basta_time_ts[dd]) for dd in range(len(tmp_basta_time_ts))]
    basta_time_dt = np.array(basta_time_dt) # holds the BASTA time array for the current file.

    #------------------------------------------------------
    # For some of the files, the date after the current one
    # holds the last hour of the day. In previous implementations,
    # we pulled in the following file only if any of the times
    # matched the current date. Now, we'll do this IN ADDITION
    # TO checking the final sounding and making sure
    # we pull in at least an hour past the final sounding.
    #------------------------------------------------------
    target_end_time = current_date_sonde_times_dt[-1]+datetime.timedelta(hours=1)
    target_start_time = current_date_sonde_times_dt[0]-datetime.timedelta(hours=1)
    
    if ii != (len(dates)-1):
        ncfile = xarray.open_dataset(basta_files[dumid+1],decode_times=False)
        after_basta_time_sec_since_00Z = np.array(ncfile['time'].copy())
        ncfile.close()
        tmp_basta_time_ts = toTimestamp(datetime.datetime(basta_dates_dt[dumid+1].year,\
                                           basta_dates_dt[dumid+1].month,\
                                           basta_dates_dt[dumid+1].day))

        tmp_basta_time_ts = tmp_basta_time_ts + after_basta_time_sec_since_00Z
        after_basta_time_dt = [toDatetime(tmp_basta_time_ts[dd]) for dd in range(len(tmp_basta_time_ts))]
        after_basta_date_dt = [datetime.datetime(after_basta_time_dt[dd].year,\
                                                after_basta_time_dt[dd].month,\
                                                after_basta_time_dt[dd].day) for dd in range(len(after_basta_time_dt))]
        after_basta_time_dt = np.array(after_basta_time_dt) # holds the BASTA time array for the after file.
        after_basta_date_dt = np.array(after_basta_date_dt) # holds the BASTA date array for the after file.
        # check to see if any of the dates in the after file equal the date on the current file
        tmpid = np.where( (after_basta_date_dt == date) | (after_basta_time_dt <= target_end_time) )

        if np.size(tmpid) > 0.:
            # now open back up after file and add indices in after file with
            # same date as current file to the current BASTA arrays
            ncfile = xarray.open_dataset(basta_files[dumid+1],decode_times=False)
            after_basta_ref = np.array(ncfile['reflectivity'].copy())
            after_basta_vel = np.array(ncfile['velocity'].copy())
            after_basta_flag = np.array(ncfile['flag'].copy())
            after_basta_flag_coupling = np.array(ncfile['flag_coupling'].copy()) # 0: no coupling (good); 1: coupling (bad)
            after_basta_noise_level = np.array(ncfile['noise_level'].copy()) # 0: good data; 1-9: bad data; -1: no data
            ncfile.close()  

            # now concatenate arrays
            basta_time_dt = np.concatenate((basta_time_dt,after_basta_time_dt[tmpid]))
            basta_ref = np.concatenate((basta_ref,np.squeeze(after_basta_ref[:,tmpid])),axis=1)
            basta_vel = np.concatenate((basta_vel,np.squeeze(after_basta_vel[:,tmpid])),axis=1)
            basta_flag = np.concatenate((basta_flag,np.squeeze(after_basta_flag[:,tmpid])),axis=1)
            basta_flag_coupling = np.concatenate((basta_flag_coupling,after_basta_flag_coupling[tmpid]),axis=0)
            basta_noise_level = np.concatenate((basta_noise_level,after_basta_noise_level[tmpid]),axis=0)

            if False:
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111)
                #levs=np.arange(-40,25,5)
                levs=np.arange(-2,2.1,0.1)
                tmpplot=ax.contourf(basta_time_dt,basta_height,basta_vel,cmap='seismic',levels=levs,extend='both')
                ax.set_ylim(0,2000)
                ax.set_xlim(datetime.datetime(2016,4,2,22,50),datetime.datetime(2016,4,2,23,10))
                plt.show()
                plt.close()
    
    # Because the current date BASTA file sometimes start at 23Z on the day prior, also need to
    # limit the current file to encompass only times on the current date (i.e., need to limit
    # the current date variables, filtering out those from 23Z-00Z on the previous date)
    basta_date_dt = np.array([datetime.datetime(basta_time_dt[dd].year,\
                                                basta_time_dt[dd].month,\
                                                basta_time_dt[dd].day) for dd in range(len(basta_time_dt))])



    basta_ref = np.squeeze(basta_ref)
    basta_vel = np.squeeze(basta_vel)
    basta_flag = np.squeeze(basta_flag)
    basta_flag_coupling = np.squeeze(basta_flag_coupling)
    basta_noise_level = np.squeeze(basta_noise_level)             

    bad_radar_data_flag = np.zeros(len(basta_time_dt))

    # create array that is the minimum detectable signal as a function of altitude
    Z_min_1km = -36.
    ref_range = 1000.
    Z_min = Z_min_1km + 20.*np.log10(basta_height) - 20.*np.log10(ref_range)
    Z_min[0] = -999.    

    # NaN out values up to 137.5 m due to surface clutter
    basta_ref[0:5,:] = np.nan
    basta_vel[0:5,:] = np.nan
    # We will also setting all basta_flag values up to 137.5 m
    # as -1. Currently basta_flag == -1 only up to 87.5 m, so we
    # want to adjust this.
    basta_flag[0:5,:] = -1

    # Set values below the minimum detectable signal to -999.
    for ttt in range(len(basta_time_dt)):
        dumid = np.where(basta_ref[:,ttt] < Z_min)
        if np.size(dumid) > 0.:
            basta_ref[dumid,ttt] = -999.
            basta_vel[dumid,ttt] = -999.
            basta_flag[dumid,ttt] = -1

    dumid = np.where(basta_flag_coupling == 1.)
    if np.size(dumid) > 0.:
        basta_ref[:,dumid] = np.nan
        basta_vel[:,dumid] = np.nan
        bad_radar_data_flag[dumid] = 1

    for ttt in range(len(basta_time_dt)):
        single_time_basta_flag = basta_flag[:,ttt]
        dumid = np.where(single_time_basta_flag > 0.)
        if np.size(dumid) > 0.:
            basta_ref[dumid,ttt] = np.nan
            basta_vel[dumid,ttt] = np.nan
        if np.all(single_time_basta_flag > 0.):
            bad_radar_data_flag[ttt] = 1    

    print('Completed radar processing.')                
    #======================================================
    # End radar block
    #======================================================   

    
    #===========================================
    # Begin ARM Ceilometer Block
    #===========================================
    tmpid = np.where(ceil_dates_dt == date)
    if np.size(tmpid) == 0.:
        ceilometer_present = False
        print('No ARM ceilometer data for this date.')
    elif np.size(tmpid) > 0.:
        print('Begin ARM ceilometer processing.')    
        
        ceilometer_present = True
        tmpid = tmpid[0][0]
        current_ceil_file = ceil_files[tmpid]
        ncfile = xarray.open_dataset(current_ceil_file,decode_times=False)
        ceil_dims = ncfile.dims
        ceil_base_time = np.array(ncfile['base_time'].copy())        
        ceil_num_times = ceil_dims['time']
        ceil_time_offset = np.array(ncfile['time_offset'].copy())
        ceil_cbh_1 = np.array(ncfile['first_cbh'].copy())
        ceil_cbh_2 = np.array(ncfile['second_cbh'].copy())
        ceil_cbh_3 = np.array(ncfile['third_cbh'].copy())
        ceil_qc_cbh_1 = np.array(ncfile['qc_first_cbh'].copy())
        ceil_qc_cbh_2 = np.array(ncfile['qc_second_cbh'].copy())
        ceil_qc_cbh_3 = np.array(ncfile['qc_third_cbh'].copy())
        ceil_status_flag = np.array(ncfile['status_flag'].copy())
        ceil_detection_status = np.array(ncfile['detection_status'].copy())
        ceil_time_ts = [int(ceil_base_time + ceil_time_offset[dd]) for dd in range(ceil_num_times)]
        ceil_time_dt = [toDatetime(ceil_time_ts[dd]) for dd in range(ceil_num_times)]    
        ceil_range_bounds = np.array(ncfile['range_bounds'].copy())
        ceil_backscatter = np.array(ncfile['backscatter'].copy())
        ceil_range = np.array(ncfile['range'].copy())
        ncfile.close()
                
        ceil_time_ts = np.array(ceil_time_ts)
        ceil_time_dt = np.array(ceil_time_dt)
        ceil_cbh_1 = np.array(ceil_cbh_1)
        ceil_cbh_2 = np.array(ceil_cbh_2)
        ceil_cbh_3 = np.array(ceil_cbh_3)
        ceil_qc_cbh_1 = np.array(ceil_qc_cbh_1)
        ceil_qc_cbh_2 = np.array(ceil_qc_cbh_2)
        ceil_qc_cbh_3 = np.array(ceil_qc_cbh_3)
        ceil_status_flag = np.array(ceil_status_flag)
        ceil_detection_status = np.array(ceil_detection_status)
        ceil_range_bounds = np.array(ceil_range_bounds)
        ceil_range = np.array(ceil_range)
        ceil_backscatter = np.array(ceil_backscatter)
        

        # pull in after file
        dumid = np.where(ceil_dates_dt == date+datetime.timedelta(days=1))
        if np.size(dumid) > 0.:
            dumid = dumid[0][0]
            after_ceil_file = ceil_files[dumid]
            ncfile = xarray.open_dataset(after_ceil_file,decode_times=False)
            after_ceil_dims = ncfile.dims
            after_ceil_base_time = np.array(ncfile['base_time'].copy())        
            after_ceil_num_times = after_ceil_dims['time']
            after_ceil_time_offset = np.array(ncfile['time_offset'].copy())
            
            after_ceil_time_ts = [int(after_ceil_base_time + after_ceil_time_offset[dd]) for dd in range(after_ceil_num_times)]
            after_ceil_time_dt = [toDatetime(after_ceil_time_ts[dd]) for dd in range(after_ceil_num_times)]              
            after_ceil_time_dt = np.array(after_ceil_time_dt)
            dumiid = np.where(after_ceil_time_dt <= target_end_time)
            if np.size(dumiid) == 0.:
                ncfile.close()
            else:
                after_ceil_cbh_1 = np.array(ncfile['first_cbh'].copy())
                after_ceil_cbh_2 = np.array(ncfile['second_cbh'].copy())
                after_ceil_cbh_3 = np.array(ncfile['third_cbh'].copy())
                after_ceil_qc_cbh_1 = np.array(ncfile['qc_first_cbh'].copy())
                after_ceil_qc_cbh_2 = np.array(ncfile['qc_second_cbh'].copy())
                after_ceil_qc_cbh_3 = np.array(ncfile['qc_third_cbh'].copy())
                after_ceil_status_flag = np.array(ncfile['status_flag'].copy())
                after_ceil_detection_status = np.array(ncfile['detection_status'].copy())\
                #after_ceil_range_bounds = np.array(ncfile['range_bounds'].copy())
                after_ceil_backscatter = np.array(ncfile['backscatter'].copy())
                #after_ceil_range = np.array(ncfile['range'].copy())
                ncfile.close()
                
                
                after_ceil_time_ts = np.array(after_ceil_time_ts)
                after_ceil_time_dt = np.array(after_ceil_time_dt)
                after_ceil_cbh_1 = np.array(after_ceil_cbh_1)
                after_ceil_cbh_2 = np.array(after_ceil_cbh_2)
                after_ceil_cbh_3 = np.array(after_ceil_cbh_3)
                after_ceil_qc_cbh_1 = np.array(after_ceil_qc_cbh_1)
                after_ceil_qc_cbh_2 = np.array(after_ceil_qc_cbh_2)
                after_ceil_qc_cbh_3 = np.array(after_ceil_qc_cbh_3)
                after_ceil_status_flag = np.array(after_ceil_status_flag)
                after_ceil_detection_status = np.array(after_ceil_detection_status)
                #after_ceil_range_bounds = np.array(after_ceil_range_bounds)
                #after_ceil_range = np.array(after_ceil_range)
                after_ceil_backscatter = np.array(after_ceil_backscatter)
                dumiid = np.squeeze(dumiid)

                ceil_time_dt = np.concatenate((ceil_time_dt,after_ceil_time_dt[dumiid]))
                ceil_time_ts = np.concatenate((ceil_time_ts,after_ceil_time_ts[dumiid]))
                ceil_cbh_1 = np.concatenate((ceil_cbh_1,after_ceil_cbh_1[dumiid]))
                ceil_cbh_2 = np.concatenate((ceil_cbh_2,after_ceil_cbh_2[dumiid]))
                ceil_cbh_3 = np.concatenate((ceil_cbh_3,after_ceil_cbh_3[dumiid]))
                ceil_qc_cbh_1 = np.concatenate((ceil_qc_cbh_1,after_ceil_qc_cbh_1[dumiid]))
                ceil_qc_cbh_2 = np.concatenate((ceil_qc_cbh_2,after_ceil_qc_cbh_2[dumiid]))
                ceil_qc_cbh_3 = np.concatenate((ceil_qc_cbh_3,after_ceil_qc_cbh_3[dumiid]))
                ceil_status_flag = np.concatenate((ceil_status_flag,after_ceil_status_flag[dumiid]))
                ceil_detection_status = np.concatenate((ceil_detection_status,after_ceil_detection_status[dumiid]))
                
                ceil_backscatter = np.concatenate((ceil_backscatter,after_ceil_backscatter[dumiid,:]))
                
        
        ceil_cbh_1_native = ceil_cbh_1.copy()
        ceil_cbh_2_native = ceil_cbh_2.copy()
        ceil_cbh_3_native = ceil_cbh_3.copy()

        #------------------------------------------
        # Interpolate ceilometer to radar time grid
        # using nearest neighbor interpolation. 
        # This method requires that the nearest
        # neighbor be within 15 seconds of the
        # radar time grid element.
        #------------------------------------------
        basta_time_ts = np.array([toTimestamp(basta_time_dt[dd]) for dd in range(len(basta_time_dt))])
        ceil_time_ts = np.array([toTimestamp(ceil_time_dt[dd]) for dd in range(len(ceil_time_dt))])
    
        basta_bin_edges = np.arange(0,np.max(basta_height)+12.5+25.,25.)
    
        ceil_cbh_1_interp = []
        ceil_cbh_2_interp = []
        ceil_cbh_3_interp = []
        ceil_detection_status_interp = []
        for ttt in range(len(basta_time_dt)):
            if bad_radar_data_flag[ttt] == 1.:
                ceil_cbh_1_interp.append(np.nan)
                ceil_cbh_2_interp.append(np.nan)
                ceil_cbh_3_interp.append(np.nan)
                ceil_detection_status_interp.append(np.nan)
                continue
            else:
                pass
            # if here, then good radar data exists
            # Now find the nearest in time ceilometer time step to the radar time step
            # If the ceilometer is more than 15 seconds away from the the radar time step,
            # then we will flag it as missing data (NaN)
            nearest_val,nearest_id = find_nearest(ceil_time_ts,basta_time_ts[ttt])
            time_diff = np.abs(nearest_val - basta_time_ts[ttt])
            target_time_diff = 15
            if time_diff <= target_time_diff:
                nearest_ceil_cbh_1 = ceil_cbh_1[nearest_id]
                nearest_ceil_cbh_2 = ceil_cbh_2[nearest_id]
                nearest_ceil_cbh_3 = ceil_cbh_3[nearest_id]
                nearest_ceil_detection_status = ceil_detection_status[nearest_id]
                ceil_detection_status_interp.append(nearest_ceil_detection_status)
                
                if np.isnan(nearest_ceil_detection_status):
                    ceil_cbh_1_interp.append(np.nan)
                    ceil_cbh_2_interp.append(np.nan)
                    ceil_cbh_3_interp.append(np.nan)
                    continue

                if np.isnan(nearest_ceil_cbh_1):
                    ceil_cbh_1_interp.append(np.nan)
                    ceil_cbh_2_interp.append(np.nan)
                    ceil_cbh_3_interp.append(np.nan)
                    continue
                    
                # ceil_cbh_1
                nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_ceil_cbh_1)
                if nearest_ceil_cbh_1 == nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_1_interp.append(midbin)
                elif nearest_ceil_cbh_1 < nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_1_interp.append(midbin)
                elif nearest_ceil_cbh_1 > nearest_val:
                    bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_1_interp.append(midbin)
                elif np.isnan(nearest_ceil_cbh_1):
                    ceil_cbh_1_interp.append(np.nan)
                    
                # ceil_cbh_2
                nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_ceil_cbh_2)
                if nearest_ceil_cbh_2 == nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_2_interp.append(midbin)
                elif nearest_ceil_cbh_2 < nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_2_interp.append(midbin)
                elif nearest_ceil_cbh_2 > nearest_val:
                    bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_2_interp.append(midbin)
                elif np.isnan(nearest_ceil_cbh_2):
                    ceil_cbh_2_interp.append(np.nan)
                    
                # ceil_cbh_3
                nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_ceil_cbh_3)
                if nearest_ceil_cbh_3 == nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_3_interp.append(midbin)
                elif nearest_ceil_cbh_3 < nearest_val:
                    bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_3_interp.append(midbin)
                elif nearest_ceil_cbh_3 > nearest_val:
                    bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                    midbin = (bin_edges[0]+bin_edges[1])/2.
                    ceil_cbh_3_interp.append(midbin)
                elif np.isnan(nearest_ceil_cbh_3):
                    ceil_cbh_3_interp.append(np.nan)
                    
            else:
                ceil_cbh_1_interp.append(np.nan)
                ceil_cbh_2_interp.append(np.nan)
                ceil_cbh_3_interp.append(np.nan)
                ceil_detection_status_interp.append(np.nan)

        ceil_cbh_1_interp = np.array(ceil_cbh_1_interp)
        ceil_cbh_2_interp = np.array(ceil_cbh_2_interp)
        ceil_cbh_3_interp = np.array(ceil_cbh_3_interp)
        ceil_detection_status_interp = np.array(ceil_detection_status_interp)
    
        print('Completed ARM ceilometer processing.')    
        
    #===========================================
    # End ARM Ceilometer Block
    #===========================================        
    
    
    #===========================================
    # Begin AAD Ceilometer Block
    #===========================================
    
    tmpid = np.where(aad_ceil_dates_dt == date)
    if np.size(tmpid) == 0.:
        aad_ceilometer_present = False
        #raise RuntimeError('No AAD files at all for this date.')
    elif np.size(tmpid) > 0.:
        print('Begin AAD ceilometer processing.')    
        
        aad_ceilometer_present = True
        current_aad_ceil_dates_dt = aad_ceil_dates_dt[tmpid]
        current_aad_ceil_times_dt = aad_ceil_times_dt[tmpid]
        current_aad_ceil_files = aad_ceil_files[tmpid]
        num_files = len(current_aad_ceil_files)
        
        aad_ceil_cbh_1 = []
        aad_ceil_cbh_2 = []
        aad_ceil_cbh_3 = []
        aad_ceil_backscatter = []
        aad_ceil_detection_status = []
        aad_ceil_time_dt = []
        aad_ceil_level = []
        aad_ceil_vertical_resolution = []
        aad_ceil_present_array = []
        
        for jj in range(num_files):
            ncfile = xarray.open_dataset(current_aad_ceil_files[jj],decode_times=False)
            ncfile_dims = ncfile.dims
            
            # in case data is missing entirely
            if np.size(ncfile.variables) <=2:
                ncfile.close()
                aad_ceil_present_array.append(False)
                continue
            
            aad_ceil_time_dims = ncfile_dims['time']
            aad_ceil_level_dims = ncfile_dims['level']
            
            # in case the time dimension is only one element long
            if aad_ceil_time_dims == 1:
                ncfile.close()
                aad_ceil_present_array.append(False)
                continue
            
            #aad_ceil_layer_dims = ncfile_dims['layer']
            tmp_aad_ceil_cbh_1 = np.array(ncfile['cbh_1'].copy())
            tmp_aad_ceil_cbh_2 = np.array(ncfile['cbh_2'].copy())
            tmp_aad_ceil_cbh_3 = np.array(ncfile['cbh_3'].copy())
            tmp_aad_ceil_backscatter = np.array(ncfile['backscatter'].copy())
            
            tmp_aad_ceil_detection_status = np.array(ncfile['detection_status'].copy())
            tmp_aad_ceil_level = np.array(ncfile['level'].copy())
            tmp_aad_ceil_vertical_resolution = np.array(ncfile['vertical_resolution'].copy())
            tmp_aad_ceil_time = np.array(ncfile['time'].copy())
            ncfile.close()
            
            # Convert times from string to datetime object
            tmp_aad_ceil_time_dt = []
            for kk in range(len(tmp_aad_ceil_time)):
                tmp_str = tmp_aad_ceil_time[kk].split('T')
                str_date = tmp_str[0]
                str_hhmmss = tmp_str[1]
                str_date = str_date.split('-')
                tmp_year = int(str_date[0])
                tmp_month = int(str_date[1])
                tmp_day = int(str_date[2])
                str_hhmmss = str_hhmmss.split(':')
                tmp_hour = int(str_hhmmss[0])
                tmp_min = int(str_hhmmss[1])
                tmp_sec = int(str_hhmmss[2])
                tmp_time = datetime.datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_min,tmp_sec)
                tmp_aad_ceil_time_dt.append(tmp_time)
                
            aad_ceil_time_dt.append(tmp_aad_ceil_time_dt)
            aad_ceil_cbh_1.append(tmp_aad_ceil_cbh_1)
            aad_ceil_cbh_2.append(tmp_aad_ceil_cbh_2)
            aad_ceil_cbh_3.append(tmp_aad_ceil_cbh_3)
            aad_ceil_backscatter.append(tmp_aad_ceil_backscatter)
            aad_ceil_detection_status.append(tmp_aad_ceil_detection_status)
            aad_ceil_level.append(tmp_aad_ceil_level)
            aad_ceil_vertical_resolution.append(tmp_aad_ceil_vertical_resolution)
            aad_ceil_present_array.append(True)
            
        if np.all(aad_ceil_present_array) != False:
            
            aad_ceil_time_dt = np.concatenate(aad_ceil_time_dt)
            aad_ceil_cbh_1 = np.concatenate(aad_ceil_cbh_1)
            aad_ceil_cbh_2 = np.concatenate(aad_ceil_cbh_2)
            aad_ceil_cbh_3 = np.concatenate(aad_ceil_cbh_3)
            aad_ceil_backscatter = np.concatenate(aad_ceil_backscatter)
            aad_ceil_detection_status = np.concatenate(aad_ceil_detection_status)
            aad_ceil_vertical_resolution = np.concatenate(aad_ceil_vertical_resolution)            
            aad_ceil_present_array = np.array(aad_ceil_present_array)
            

        # after files
        dumid = np.where(aad_ceil_dates_dt == date+datetime.timedelta(days=1))
        if np.size(dumid) > 0.:
            after_aad_ceilometer_present = True
            after_aad_ceil_dates_dt = aad_ceil_dates_dt[dumid]
            after_aad_ceil_times_dt = aad_ceil_times_dt[dumid]
            after_aad_ceil_files = aad_ceil_files[dumid]

            dumid2 = np.where(after_aad_ceil_times_dt <= target_end_time)
            if np.size(dumid2) > 0.:

                after_aad_ceil_dates_dt = aad_ceil_dates_dt[dumid2]
                after_aad_ceil_times_dt = aad_ceil_times_dt[dumid2]
                after_aad_ceil_files = aad_ceil_files[dumid2]
                num_after_files = len(after_aad_ceil_files)

                after_aad_ceil_cbh_1 = []
                after_aad_ceil_cbh_2 = []
                after_aad_ceil_cbh_3 = []
                after_aad_ceil_backscatter = []
                after_aad_ceil_detection_status = []
                after_aad_ceil_time_dt = []
                after_aad_ceil_level = []
                after_aad_ceil_vertical_resolution = []
                after_aad_ceil_present_array = []

                for jj in range(num_after_files):
                    ncfile = xarray.open_dataset(after_aad_ceil_files[jj],decode_times=False)
                    ncfile_dims = ncfile.dims

                    # in case data is missing entirely
                    if np.size(ncfile.variables) <=2:
                        ncfile.close()
                        after_aad_ceil_present_array.append(False)
                        continue

                    after_aad_ceil_time_dims = ncfile_dims['time']
                    after_aad_ceil_level_dims = ncfile_dims['level']

                    # in case the time dimension is only one element long
                    if after_aad_ceil_time_dims == 1:
                        ncfile.close()
                        after_aad_ceil_present_array.append(False)
                        continue

                    #aad_ceil_layer_dims = ncfile_dims['layer']
                    tmp_after_aad_ceil_cbh_1 = np.array(ncfile['cbh_1'].copy())
                    tmp_after_aad_ceil_cbh_2 = np.array(ncfile['cbh_2'].copy())
                    tmp_after_aad_ceil_cbh_3 = np.array(ncfile['cbh_3'].copy())
                    tmp_after_aad_ceil_backscatter = np.array(ncfile['backscatter'].copy())
                    tmp_after_aad_ceil_detection_status = np.array(ncfile['detection_status'].copy())
                    tmp_after_aad_ceil_level = np.array(ncfile['level'].copy())
                    tmp_after_aad_ceil_vertical_resolution = np.array(ncfile['vertical_resolution'].copy())
                    tmp_after_aad_ceil_time = np.array(ncfile['time'].copy())
                    ncfile.close()

                    # Convert times from string to datetime object
                    tmp_after_aad_ceil_time_dt = []
                    for kk in range(len(tmp_after_aad_ceil_time)):
                        tmp_str = tmp_after_aad_ceil_time[kk].split('T')
                        str_date = tmp_str[0]
                        str_hhmmss = tmp_str[1]
                        str_date = str_date.split('-')
                        tmp_year = int(str_date[0])
                        tmp_month = int(str_date[1])
                        tmp_day = int(str_date[2])
                        str_hhmmss = str_hhmmss.split(':')
                        tmp_hour = int(str_hhmmss[0])
                        tmp_min = int(str_hhmmss[1])
                        tmp_sec = int(str_hhmmss[2])
                        tmp_time = datetime.datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_min,tmp_sec)
                        tmp_after_aad_ceil_time_dt.append(tmp_time)

                    after_aad_ceil_time_dt.append(tmp_after_aad_ceil_time_dt)
                    after_aad_ceil_cbh_1.append(tmp_after_aad_ceil_cbh_1)
                    after_aad_ceil_cbh_2.append(tmp_after_aad_ceil_cbh_2)
                    after_aad_ceil_cbh_3.append(tmp_after_aad_ceil_cbh_3)
                    after_aad_ceil_backscatter.append(tmp_after_aad_ceil_backscatter)
                    after_aad_ceil_detection_status.append(tmp_after_aad_ceil_detection_status)
                    after_aad_ceil_level.append(tmp_after_aad_ceil_level)
                    after_aad_ceil_vertical_resolution.append(tmp_after_aad_ceil_vertical_resolution)
                    after_aad_ceil_present_array.append(True)

                if np.size(after_aad_ceil_present_array) != 0.:
                    

                    if np.all(after_aad_ceil_present_array) != False:

                        after_aad_ceil_time_dt = np.concatenate(after_aad_ceil_time_dt)
                        after_aad_ceil_cbh_1 = np.concatenate(after_aad_ceil_cbh_1)
                        after_aad_ceil_cbh_2 = np.concatenate(after_aad_ceil_cbh_2)
                        after_aad_ceil_cbh_3 = np.concatenate(after_aad_ceil_cbh_3)
                        after_aad_ceil_backscatter = np.concatenate(after_aad_ceil_backscatter)
                        after_aad_ceil_detection_status = np.concatenate(after_aad_ceil_detection_status)
                        after_aad_ceil_vertical_resolution = np.concatenate(after_aad_ceil_vertical_resolution)
                        after_aad_ceil_present_array = np.array(after_aad_ceil_present_array)

                        dumiid = np.where(after_aad_ceil_time_dt <= target_end_time)
                        if np.size(dumiid) > 0.:
                            dumiid = np.squeeze(dumiid)
                            after_aad_ceil_time_dt = after_aad_ceil_time_dt[dumiid]
                            after_aad_ceil_cbh_1 = after_aad_ceil_cbh_1[dumiid]
                            after_aad_ceil_cbh_2 = after_aad_ceil_cbh_2[dumiid]
                            after_aad_ceil_cbh_3 = after_aad_ceil_cbh_3[dumiid]
                            after_aad_ceil_vertical_resolution = after_aad_ceil_vertical_resolution[dumiid]
                            after_aad_ceil_detection_status = after_aad_ceil_detection_status[dumiid]
                            after_aad_ceil_backscatter = after_aad_ceil_backscatter[dumiid,:]
                            aad_ceil_time_dt = np.concatenate((aad_ceil_time_dt,after_aad_ceil_time_dt))
                            aad_ceil_cbh_1 = np.concatenate((aad_ceil_cbh_1,after_aad_ceil_cbh_1))
                            aad_ceil_cbh_2 = np.concatenate((aad_ceil_cbh_2,after_aad_ceil_cbh_2))
                            aad_ceil_cbh_3 = np.concatenate((aad_ceil_cbh_3,after_aad_ceil_cbh_3))
                            aad_ceil_vertical_resolution = np.concatenate((aad_ceil_vertical_resolution,after_aad_ceil_vertical_resolution))
                            aad_ceil_detection_status = np.concatenate((aad_ceil_detection_status,after_aad_ceil_detection_status))
                            aad_ceil_backscatter = np.concatenate((aad_ceil_backscatter,after_aad_ceil_backscatter))
                            aad_ceil_present_array = np.concatenate((aad_ceil_present_array,after_aad_ceil_present_array))
        
        if np.all(aad_ceil_present_array) != False:    
            
            aad_ceil_level = aad_ceil_level[0]
            max_cbh_arm_ceil = 7430. # m
            
            # Limit max CBH to the maximum detectable by the ARM ceilometer
            tmpid = np.where(aad_ceil_cbh_1 > max_cbh_arm_ceil)
            if np.size(tmpid) > 0.:
                aad_ceil_cbh_1[tmpid] = np.nan     
                
            # Limit max CBH to the maximum detectable by the ARM ceilometer
            tmpid = np.where(aad_ceil_cbh_2 > max_cbh_arm_ceil)
            if np.size(tmpid) > 0.:
                aad_ceil_cbh_2[tmpid] = np.nan   
                
            # Limit max CBH to the maximum detectable by the ARM ceilometer
            tmpid = np.where(aad_ceil_cbh_3 > max_cbh_arm_ceil)
            if np.size(tmpid) > 0.:
                aad_ceil_cbh_3[tmpid] = np.nan   

            # ensure that the vertical resolution is always 10
            unique_res = np.unique(aad_ceil_vertical_resolution)
            if np.size(unique_res) > 1.:
                raise RuntimeError("AAD ceilometer resolution is not unique.")
            aad_ceil_height = (aad_ceil_level+1)*unique_res[0] 

            
            #------------------------------------------
            # Interpolate ceilometer to radar time grid
            # using nearest neighbor interpolation. 
            # This method requires that the nearest
            # neighbor be within 15 seconds of the
            # radar time grid element.
            #------------------------------------------
            basta_time_ts = np.array([toTimestamp(basta_time_dt[dd]) for dd in range(len(basta_time_dt))])
            aad_ceil_time_ts = np.array([toTimestamp(aad_ceil_time_dt[dd]) for dd in range(len(aad_ceil_time_dt))])

            basta_bin_edges = np.arange(0,np.max(basta_height)+12.5+25.,25.)

            aad_ceil_cbh_1_interp = []
            aad_ceil_cbh_2_interp = []
            aad_ceil_cbh_3_interp = []
            aad_ceil_detection_status_interp = []
            for ttt in range(len(basta_time_dt)):
                if bad_radar_data_flag[ttt] == 1.:
                    aad_ceil_cbh_1_interp.append(np.nan)
                    aad_ceil_cbh_2_interp.append(np.nan)
                    aad_ceil_cbh_3_interp.append(np.nan)
                    aad_ceil_detection_status_interp.append(np.nan)
                    continue
                else:
                    pass
                # if here, then good radar data exists
                # Now find the nearest in time ceilometer time step to the radar time step
                # If the ceilometer is more than 15 seconds away from the the radar time step,
                # then we will flag it as missing data (NaN)
                nearest_val,nearest_id = find_nearest(aad_ceil_time_ts,basta_time_ts[ttt])
                time_diff = np.abs(nearest_val - basta_time_ts[ttt])
                target_time_diff = 15
                if time_diff <= target_time_diff:
                    nearest_aad_ceil_cbh_1 = aad_ceil_cbh_1[nearest_id]
                    nearest_aad_ceil_cbh_2 = aad_ceil_cbh_2[nearest_id]
                    nearest_aad_ceil_cbh_3 = aad_ceil_cbh_3[nearest_id]
                    nearest_aad_ceil_detection_status = aad_ceil_detection_status[nearest_id]
                    dum = nearest_aad_ceil_detection_status.decode('UTF-8')
                    dum = int(dum)
                    aad_ceil_detection_status_interp.append(dum)


                    if np.isnan(nearest_aad_ceil_cbh_1):
                        aad_ceil_cbh_1_interp.append(np.nan)
                        aad_ceil_cbh_2_interp.append(np.nan)
                        aad_ceil_cbh_3_interp.append(np.nan)
                        continue

                    # ceil_cbh_1
                    nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_aad_ceil_cbh_1)
                    if nearest_aad_ceil_cbh_1 == nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_1_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_1 < nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_1_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_1 > nearest_val:
                        bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_1_interp.append(midbin)
                    elif np.isnan(nearest_ceil_cbh_1):
                        aad_ceil_cbh_1_interp.append(np.nan)
                        
                    # ceil_cbh_2
                    nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_aad_ceil_cbh_2)
                    if nearest_aad_ceil_cbh_2 == nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_2_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_2 < nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_2_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_2 > nearest_val:
                        bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_2_interp.append(midbin)
                    elif np.isnan(nearest_aad_ceil_cbh_2):
                        aad_ceil_cbh_2_interp.append(np.nan)
                        
                    # ceil_cbh_3
                    nearest_val,nearest_id = find_nearest(basta_bin_edges,nearest_aad_ceil_cbh_3)
                    if nearest_aad_ceil_cbh_3 == nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_3_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_3 < nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_3_interp.append(midbin)
                    elif nearest_aad_ceil_cbh_3 > nearest_val:
                        bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        aad_ceil_cbh_3_interp.append(midbin)
                    elif np.isnan(nearest_aad_ceil_cbh_3):
                        aad_ceil_cbh_3_interp.append(np.nan)
                        
                    else:
                        print(aaaaa)
                else:
                    aad_ceil_cbh_1_interp.append(np.nan)
                    aad_ceil_cbh_2_interp.append(np.nan)
                    aad_ceil_cbh_3_interp.append(np.nan)
                    aad_ceil_detection_status_interp.append(np.nan)

            aad_ceil_cbh_1_interp = np.array(aad_ceil_cbh_1_interp)
            aad_ceil_cbh_2_interp = np.array(aad_ceil_cbh_2_interp)
            aad_ceil_cbh_3_interp = np.array(aad_ceil_cbh_3_interp)
            aad_ceil_detection_status_interp = np.array(aad_ceil_detection_status_interp)

            
            print('End AAD ceilometer processing.')    
                        
        else:

            aad_ceilometer_present = False
        

    #===========================================
    # End AAD Ceilometer Block
    #===========================================
    if (ceilometer_present == False) and (aad_ceilometer_present == False):
        print('No ARM nor AAD ceilometer data for this date. Ommitting date.')
        ii+=1
        continue

        
    #===========================================
    # Begin Merged Ceilometer Block
    #===========================================
    # Use the ARM ceilometer as the primary ceilometer unless a file doesn't 
    # exist for the current date, in which case we'll use the AAD ceilometer
    # as the primary ceilometer.
    print('Begin merged ceilometer processing.') 
    
    
    if ceilometer_present == True:
        # NaN out values
        dumid = np.where( (np.isnan(ceil_detection_status_interp)) | (ceil_detection_status_interp == 0) | (ceil_detection_status_interp > 3))
        if np.size(dumid) > 0.:
            ceil_cbh_1_interp[dumid] = np.nan
            ceil_cbh_2_interp[dumid] = np.nan
            ceil_cbh_3_interp[dumid] = np.nan
            
    if aad_ceilometer_present == True:
        # NaN out values
        dumid = np.where( (np.isnan(aad_ceil_detection_status_interp)) | (aad_ceil_detection_status_interp == 0.) | (aad_ceil_detection_status_interp > 3.))
        if np.size(dumid) > 0.:
            aad_ceil_cbh_1_interp[dumid] = np.nan
            aad_ceil_cbh_2_interp[dumid] = np.nan
            aad_ceil_cbh_3_interp[dumid] = np.nan

    
    if (ceilometer_present == True) and (aad_ceilometer_present == True):
        tmpid = np.where(np.isnan(ceil_cbh_1_interp) & ~np.isnan(aad_ceil_cbh_1_interp))
        merged_ceil_cbh_1 = ceil_cbh_1_interp.copy()
        merged_ceil_cbh_2 = ceil_cbh_2_interp.copy()
        merged_ceil_cbh_3 = ceil_cbh_3_interp.copy()
        if np.size(tmpid) > 0.:
            merged_ceil_cbh_1[tmpid] = aad_ceil_cbh_1_interp[tmpid]
            merged_ceil_cbh_2[tmpid] = aad_ceil_cbh_2_interp[tmpid]
            merged_ceil_cbh_3[tmpid] = aad_ceil_cbh_3_interp[tmpid]
    elif (ceilometer_present == False) and (aad_ceilometer_present == True):
        merged_ceil_cbh_1 = aad_ceil_cbh_1_interp
        merged_ceil_cbh_2 = aad_ceil_cbh_2_interp
        merged_ceil_cbh_3 = aad_ceil_cbh_3_interp
    elif (ceilometer_present == True) and (aad_ceilometer_present == False):
        merged_ceil_cbh_1 = ceil_cbh_1_interp
        merged_ceil_cbh_2 = ceil_cbh_2_interp
        merged_ceil_cbh_3 = ceil_cbh_3_interp
    elif (ceilometer_present == False) and (aad_ceilometer_present == False):
        merged_ceil_cbh_1 = np.nan
        merged_ceil_cbh_2 = np.nan
        merged_ceil_cbh_3 = np.nan
        

    print('Completed merged ceilometer processing.')    
    
    #===========================================
    # End Merged Ceilometer Block
    #===========================================    
                
        
        

    
    #===========================================
    # Begin Sonde Interpolation
    #===========================================   
    print('Begin sounding interpolation.')    
    # Before interpolation, check maximum altitude of all soundings
            
    # height interpolation to basta grid
    sonde_temperature_interp = []
    sonde_pressure_interp = []
    sonde_rh_interp = []
    sonde_rh_i_interp = []
    sonde_q_interp = []
    sonde_u_interp = []
    sonde_v_interp = []
    sonde_wind_dir_interp = []
    sonde_wind_speed_interp = []
    sonde_theta_interp = []
    sonde_theta_e_interp = []
    for jj in range(len(sonde_time_dt)):
        sonde_pressure_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_pressure[jj]))
        sonde_temperature_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_temperature[jj]))
        sonde_q_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_q[jj]))
        sonde_rh_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_rh[jj]))
        sonde_rh_i_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_rh_i[jj]))
        sonde_u_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_u[jj]))
        sonde_v_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_v[jj]))
        sonde_wind_speed_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_wind_speed[jj]))
        #sonde_wind_direction_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_wind_dir[jj]))
        # wind direction needs to use nearest neighbor interpolation
        #tmp_sonde_wind_dir_interp = []
        #for kk in range(len(basta_height)):
        #    nearest_wind_direction,nearest_id = find_nearest(sonde_height[jj]*1.e3,basta_height[kk])
        #    tmp_sonde_wind_dir = sonde_wind_dir[jj]
        #    tmp_sonde_wind_dir_interp.append(tmp_sonde_wind_dir[nearest_id])
        #tmp_sonde_wind_dir_interp = np.array(tmp_sonde_wind_dir_interp)
        sonde_wind_dir_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_wind_dir[jj],period=360))
        sonde_theta_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_theta[jj]))
        sonde_theta_e_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_theta_e[jj]))
            

    sonde_temperature_interp = np.array(sonde_temperature_interp)
    sonde_pressure_interp = np.array(sonde_pressure_interp)
    sonde_q_interp = np.array(sonde_q_interp)
    sonde_rh_interp = np.array(sonde_rh_interp)
    sonde_rh_i_interp = np.array(sonde_rh_i_interp)
    sonde_u_interp = np.array(sonde_u_interp)
    sonde_v_interp = np.array(sonde_v_interp)
    sonde_wind_speed_interp = np.array(sonde_wind_speed_interp)
    sonde_wind_dir_interp = np.array(sonde_wind_dir_interp)
    sonde_theta_interp = np.array(sonde_theta_interp)
    sonde_theta_e_interp = np.array(sonde_theta_e_interp)
    
    print('Completed sounding interpolation.')    
    #===========================================
    # End Sonde Interpolation
    #===========================================  
    
    
    
    

    
    #===========================================
    # Begin Surface Meteorology Block
    #===========================================
    basta_time_ts = np.array([toTimestamp(basta_time_dt[dd]) for dd in range(len(basta_time_dt))])

    tmpid = np.where(sfc_dates_dt == date)
    if np.size(tmpid) == 0.:
        sfc_present = False
    elif np.size(tmpid) > 0.:
        print('Begin surface meteorology processing.')    
        
        sfc_present = True
        tmpid = tmpid[0][0]
        current_sfc_file = sfc_files[tmpid]
        ncfile = xarray.open_dataset(current_sfc_file,decode_times=False)
        sfc_dims = ncfile.dims
        sfc_base_time = np.array(ncfile['base_time'].copy())        
        sfc_num_times = sfc_dims['time']
        sfc_time_offset = np.array(ncfile['time_offset'].copy())
        sfc_temperature = np.array(ncfile['temperature'].copy())
        sfc_pressure = np.array(ncfile['pressure'].copy())
        sfc_rh = np.array(ncfile['relative_humidity'].copy())
        sfc_wind_speed = np.array(ncfile['wind_speed'].copy())
        sfc_wind_dir = np.array(ncfile['wind_direction'].copy())
        sfc_precip = np.array(ncfile['cumulative_precipitation'].copy())        
        ncfile.close()
        
        sfc_time_ts = [int(sfc_base_time + sfc_time_offset[dd]) for dd in range(sfc_num_times)]
        sfc_time_dt = [toDatetime(sfc_time_ts[dd]) for dd in range(sfc_num_times)]        

        dumid = np.where(sfc_dates_dt == date+datetime.timedelta(days=1))
        if np.size(dumid) > 0.:
            dumid = dumid[0][0]
        
            after_sfc_file = sfc_files[dumid]
            ncfile = xarray.open_dataset(after_sfc_file,decode_times=False)
            after_sfc_dims = ncfile.dims
            after_sfc_base_time = np.array(ncfile['base_time'].copy())        
            after_sfc_num_times = after_sfc_dims['time']
            after_sfc_time_offset = np.array(ncfile['time_offset'].copy())
            after_sfc_temperature = np.array(ncfile['temperature'].copy())
            after_sfc_pressure = np.array(ncfile['pressure'].copy())
            after_sfc_rh = np.array(ncfile['relative_humidity'].copy())
            after_sfc_wind_speed = np.array(ncfile['wind_speed'].copy())
            after_sfc_wind_dir = np.array(ncfile['wind_direction'].copy())
            after_sfc_precip = np.array(ncfile['cumulative_precipitation'].copy())        
            ncfile.close()
            
            after_sfc_time_ts = [int(after_sfc_base_time + after_sfc_time_offset[dd]) for dd in range(after_sfc_num_times)]
            after_sfc_time_dt = [toDatetime(after_sfc_time_ts[dd]) for dd in range(after_sfc_num_times)]             
            after_sfc_time_ts = np.array(after_sfc_time_ts)
            after_sfc_time_dt = np.array(after_sfc_time_dt)
            
            dumiid = np.where(after_sfc_time_dt <= target_end_time)
            if np.size(dumiid) > 0.:
                dumiid = np.squeeze(dumiid)
                sfc_time_ts = np.concatenate((sfc_time_ts,after_sfc_time_ts[dumiid]))
                sfc_time_dt = np.concatenate((sfc_time_dt,after_sfc_time_dt[dumiid]))
                sfc_temperature = np.concatenate((sfc_temperature,after_sfc_temperature[dumiid]))
                sfc_pressure = np.concatenate((sfc_pressure,after_sfc_pressure[dumiid]))
                sfc_rh = np.concatenate((sfc_rh,after_sfc_rh[dumiid]))
                sfc_wind_speed = np.concatenate((sfc_wind_speed,after_sfc_wind_speed[dumiid]))
                sfc_wind_dir = np.concatenate((sfc_wind_dir,after_sfc_wind_dir[dumiid]))
                sfc_precip = np.concatenate((sfc_precip,after_sfc_precip[dumiid]))
            
        
        # NaN out missing values
        # original missing value is -9999.
        sfc_temperature[sfc_temperature < -950.] = np.nan
        sfc_pressure[sfc_pressure < -950.] = np.nan
        sfc_rh[sfc_rh < -950.] = np.nan
        sfc_wind_speed[sfc_wind_speed < -950.] = np.nan
        sfc_wind_dir[sfc_wind_dir < -950.] = np.nan
        sfc_precip[sfc_precip < -950.] = np.nan

        # interpolate sfc variables to basta time grid
        sfc_temperature_interp = np.interp(basta_time_ts,sfc_time_ts,sfc_temperature)
        sfc_rh_interp = np.interp(basta_time_ts,sfc_time_ts,sfc_rh)
        sfc_wind_speed_interp = np.interp(basta_time_ts,sfc_time_ts,sfc_wind_speed)
        sfc_wind_dir_interp = np.interp(basta_time_ts,sfc_time_ts,sfc_wind_dir,period=360)
        sfc_pressure_interp = np.interp(basta_time_ts,sfc_time_ts,sfc_pressure)

        sfc_time_dt = np.array(sfc_time_dt)
        sfc_temperature = np.array(sfc_temperature)
        sfc_rh = np.array(sfc_rh)
        sfc_pressure = np.array(sfc_pressure)        
        
        
        print('End surface meteorology processing.')    
        
    #===========================================
    # End Surface Meteorology Block
    #===========================================    

    #===========================================
    # Begin Cluster ID block
    #===========================================
    if False:
        print('Begin cluster processing.')    
    
        current_cluster_times_dt = []
        current_cluster_dates_dt = []
        current_cluster_ids = []
        current_cluster_present = []
        for jj in range(len(sonde_time_dt)):
            tmpid = np.where(cluster_times_dt == sonde_time_dt[jj])
            if np.size(tmpid) > 0.:
                current_cluster_times_dt.append(cluster_times_dt[tmpid])
                current_cluster_dates_dt.append(cluster_dates_dt[tmpid])
                current_cluster_ids.append(cluster_id[tmpid][0])
                current_cluster_present.append(1)
            else:
                current_cluster_present.append(0)
                current_cluster_times_dt.append(-999.)
                current_cluster_dates_dt.append(-999.)
                current_cluster_ids.append(-999.)

        current_cluster_present = np.array(current_cluster_present)
        current_cluster_ids = np.array(current_cluster_ids)
        current_cluster_times_dt = np.array(current_cluster_times_dt)
        current_cluster_dates_dt = np.array(current_cluster_dates_dt)
        print('Completed cluster processing.')    
    #===========================================
    # End Cluster ID block
    #=========================================== 

    #===========================================
    # Begin Satellite block
    #===========================================
    print('Begin satellite processing.')  
    tmpid = np.where(sat_dates_dt == date)[0][0]
    
    ncfile = xarray.open_dataset(sat_files[tmpid],decode_times=False)
    sat_time_epoch = np.array(ncfile['time_epoch'].copy())
    sat_time_dt = np.array([toDatetime(sat_time_epoch[dd]) for dd in range(len(sat_time_epoch))])
    sat_vis = ncfile['visible_reflectance'].copy()
    sat_irtb = ncfile['ir_brightness_temperature'].copy()
    sat_lat = ncfile['lat'].copy()
    sat_lon = ncfile['lon'].copy()
    ncfile.close()
    
    tmpid_before = tmpid-1
    ncfile = xarray.open_dataset(sat_files[tmpid_before],decode_times=False)
    before_sat_time_epoch = np.array(ncfile['time_epoch'].copy())
    before_sat_time_dt = np.array([toDatetime(before_sat_time_epoch[dd]) for dd in range(len(before_sat_time_epoch))])
    before_sat_vis = ncfile['visible_reflectance'].copy()
    before_sat_irtb = ncfile['ir_brightness_temperature'].copy()
    ncfile.close()

    tmpid_after = tmpid+1
    ncfile = xarray.open_dataset(sat_files[tmpid_after],decode_times=False)
    after_sat_time_epoch = np.array(ncfile['time_epoch'].copy())
    after_sat_time_dt = np.array([toDatetime(after_sat_time_epoch[dd]) for dd in range(len(after_sat_time_epoch))])
    after_sat_vis = ncfile['visible_reflectance'].copy()
    after_sat_irtb = ncfile['ir_brightness_temperature'].copy()
    ncfile.close()    
    
    sat_time_dt_all = np.concatenate([before_sat_time_dt,sat_time_dt,after_sat_time_dt])
    sat_vis_all = np.concatenate([before_sat_vis,sat_vis,after_sat_vis],axis=2)
    sat_irtb_all = np.concatenate([before_sat_irtb,sat_irtb,after_sat_irtb],axis=2)
    sat_time_ts_all = np.array([toTimestamp(sat_time_dt_all[dd]) for dd in range(len(sat_time_dt_all))])
    sat_vis_all[sat_vis_all > 1.] = 1.
        
    print('Completed satellite processing.')    
    #===========================================
    # End Satellite block
    #===========================================   
    
    #===========================================
    # Start Plot
    #===========================================        
    #for jj in range(num_current_soundings):
    for jj in range(1,2):
        #print(jj)
        time_delta = datetime.timedelta(hours=1)
        start_time = sonde_time_dt_long[jj][0]-time_delta
        end_time = sonde_time_dt_long[jj][0]+time_delta
        
        dumid = np.where((basta_time_dt >= start_time) & (basta_time_dt <= end_time))
        basta_time_lim = basta_time_dt[dumid]

        if np.size(basta_time_lim) == 0.:
            basta_present_flag = False
        else:
            basta_present_flag = True

        print('Sonde time: {}'.format(sonde_time_dt_long[jj][0]))
        print('Before target time: {}'.format(start_time))
        print('After target time: {}'.format(end_time))
        
        sat_target_time_dt = sonde_time_dt_long[jj][0]
        sat_target_time_ts = toTimestamp(sat_target_time_dt)
        nearest_val,nearest_id = find_nearest(sat_time_ts_all,sat_target_time_ts)
        print('Nearest satellite time: {}'.format(toDatetime(nearest_val)))
        sat_vis_1 = sat_vis_all[:,:,nearest_id]
        sat_irtb_1 = sat_irtb_all[:,:,nearest_id]
        sat_nearest_time = toDatetime(nearest_val)

        before_sat_target_time_dt = start_time
        before_sat_target_time_ts = toTimestamp(before_sat_target_time_dt)
        nearest_val,nearest_id = find_nearest(sat_time_ts_all,before_sat_target_time_ts)
        print('Nearest satellite before time: {}'.format(toDatetime(nearest_val)))
        sat_vis_before = sat_vis_all[:,:,nearest_id]
        sat_irtb_before = sat_irtb_all[:,:,nearest_id] 
        sat_before_nearest_time = toDatetime(nearest_val)
        
        
        after_sat_target_time_dt = end_time
        after_sat_target_time_ts = toTimestamp(after_sat_target_time_dt)
        nearest_val,nearest_id = find_nearest(sat_time_ts_all,after_sat_target_time_ts)
        print('Nearest satellite after time: {}'.format(toDatetime(nearest_val)))
        sat_vis_after = sat_vis_all[:,:,nearest_id]
        sat_irtb_after = sat_irtb_all[:,:,nearest_id]        
        sat_after_nearest_time = toDatetime(nearest_val)
                        
        # Find cloud layers in RH profile
        #current_sonde_rh = sonde_rh_interp[jj]
        current_sonde_rh = sonde_rh_interp[jj]
        # mask RH
        cloud_layer_mask = np.zeros(len(current_sonde_rh))

        dumid = np.where(current_sonde_rh >= 95.)
        if np.size(dumid) > 0.:
            cloud_layer_mask[dumid] = 1

        #cloud_layer_mask[np.isnan(cloud_layer_mask)] = 0
        cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)
        new_cloud_layer_mask = cloud_layer_mask.copy()
        thresh_thick = 25.
        # concatenate cloud layers
        if num_cloud_objects > 1:
            alt_diff = np.zeros(num_cloud_objects-1)

            unique_ids = np.unique(cloud_Objects)
            unique_ids = unique_ids[1:]

            for kk in range(len(unique_ids)-1):
                ids_layer_1 = np.where(cloud_Objects == kk+1)
                ids_layer_1 = ids_layer_1[0]
                ids_layer_2 = np.where(cloud_Objects == kk+2)
                ids_layer_2 = ids_layer_2[0]

                if np.size(ids_layer_1) == 1:
                    layer_1_top = ids_layer_1[0]
                else:
                    layer_1_top = ids_layer_1[-1]

                if np.size(ids_layer_2) == 1:
                    layer_2_bottom = ids_layer_2[0]
                else:
                    layer_2_bottom = ids_layer_2[0]

                alt_diff[kk] =  (layer_2_bottom-layer_1_top)*25.     



            # create a new cloud mask that accounts for concatenating layers
            for kk in range(len(alt_diff)):
                if alt_diff[kk] <= (thresh_thick+25.):
                    ids_layer_1 = np.where(cloud_Objects == kk+1)
                    ids_layer_1 = ids_layer_1[0]
                    ids_layer_2 = np.where(cloud_Objects == kk+2)
                    ids_layer_2 = ids_layer_2[0]
                    new_cloud_layer_mask[ids_layer_1[0]:ids_layer_2[-1]+1] = 1
                    
        # now redo object identification
        cloud_Objects,num_cloud_objects = ndimage.label(new_cloud_layer_mask)             
        

        cloud_layer_mask = new_cloud_layer_mask
        sonde_cbhs = []
        sonde_cths = []
        sonde_c_thicks = []
        if num_cloud_objects > 0.:
            for kk in range(num_cloud_objects):
                dumid = np.where(cloud_Objects == kk+1)
                tmp_cbh = basta_height[dumid[0][0]]
                tmp_cth = basta_height[dumid[0][-1]]
                tmp_c_thick = tmp_cth - tmp_cbh
                if tmp_c_thick > 25.:
                    sonde_c_thicks.append(tmp_c_thick)
                    sonde_cbhs.append(tmp_cbh)
                    sonde_cths.append(tmp_cth)
        num_cloud_objects = len(sonde_cbhs)
        
        dfmt = mdates.DateFormatter('%H:%M')

        if basta_present_flag == True:
            dumid = np.where((basta_time_dt >= start_time) & (basta_time_dt <= end_time))[0]
            basta_ref_lim = basta_ref[:,dumid]
            basta_vel_lim = basta_vel[:,dumid]
            basta_time_lim = basta_time_dt[dumid]
            merged_ceil_cbh_1_lim = merged_ceil_cbh_1[dumid]
            merged_ceil_cbh_2_lim = merged_ceil_cbh_2[dumid]
            merged_ceil_cbh_3_lim = merged_ceil_cbh_3[dumid]



            # calculate cloud fraction to determine height limit.    
            cloud_frac = []
            for kk in range(len(basta_height)):
                if kk < 5:
                    cloud_frac.append(0)
                    continue
                else:
                    pass
                dumid = np.where(~np.isnan(basta_ref_lim[kk,:]) & (basta_ref_lim[kk,:] > -999.))
                if np.size(dumid) > 0.:
                    tmp_cloud_frac = np.size(dumid)/np.size(basta_time_lim)
                    cloud_frac.append(tmp_cloud_frac)
                else:
                    cloud_frac.append(0)
            cloud_frac = np.array(cloud_frac)
            
            dumid = np.where(sonde_rh_interp[jj] >= 98.)
            if np.size(dumid) > 0.:
                dumid = dumid[0]
                cloud_frac[dumid] = 1.

            if np.max(cloud_frac) == 0.:
                continue
            dumid = np.where(cloud_frac > 0.1)
            if np.size(dumid) > 0.:
                max_height = (basta_height[dumid[0][-1]]+100.)/1.e3
            else:
                dumid = np.where(cloud_frac > 0.)
                max_height = (basta_height[dumid[0][-1]]+100.)/1.e3
                
            #cloud_mask = np.zeros(np.size(cloud_frac))
            #cloud_mask[cloud_frac > 0.] = 1
            #cloud_objects,num_cloud_objects = ndimage.label(cloud_mask)
            #for dd in range(num_cloud_objects):
            #    tmpid = np.where(cloud_objects == dd+1)
            #    dumsize = np.size(tmpid)
            #dumid = np.where(cloud_mask == 1.)
            #max_height = (basta_height[dumid[0][-1]]+100.)/1.e3        

            from scipy import stats
            if max_height < 1.:
                dum_max = stats.mode(merged_ceil_cbh_1_lim[~np.isnan(merged_ceil_cbh_1_lim)])
                if np.size(dum_max) > 0.:
                    dum_max = dum_max[0][0]
                    dum_max = dum_max/1.e3 + 0.25
                    if dum_max > 1.:
                        max_height = dum_max
                    else:
                        max_height = 1.
                else:
                    max_height = 1.
                
            y_height = max_height
            #y_height = 2.
            y_height_dum,y_height_id = find_nearest(sonde_height[jj],y_height)
            y_height = y_height_const
            dumid = np.where(basta_height <= y_height*1.e3)[0]
            basta_ref_lim = basta_ref_lim[dumid,:]
            basta_vel_lim = basta_vel_lim[dumid,:]
            basta_height_lim = basta_height[dumid]
            #basta_ref_lim[basta_ref_lim == -999.] = np.nan
            #basta_vel_lim[basta_vel_lim == -999.] = np.nan 
            

        else:
            y_height = 10.
            y_height_dum,y_height_id = find_nearest(sonde_height[jj],y_height)
        
        #--------------------
        # Plot
        #--------------------
        sns.set_theme()
        #sns.set_style('dark')
        sns.set_style('ticks')
        sns.set(rc={'axes.facecolor':'white','axes.edgecolor': 'black','grid.color':'grey'})
        sns.set_context('talk')        
        
        
        Fontsize=15
        fig = plt.figure(figsize=(12,18))  
        gs=GridSpec(4,3) # 4 rows, 3 columns
        ax1 = fig.add_subplot(gs[0,0])
        ax1a = ax1.twiny()
        ax2 = fig.add_subplot(gs[0,1])       
        ax3 = fig.add_subplot(gs[0,2])       
        ax4 = fig.add_subplot(gs[1,:])  
        ax5 = fig.add_subplot(gs[2,:]) 
        ax6 = fig.add_subplot(gs[3,:])  
        
        axlist = [ax1,ax2,ax4,ax5,ax6]
        for ax in axlist:
            ax.tick_params(labelsize=Fontsize)
           # ax.grid(which='both',c='grey',axis='both')
            
        axlist2 = [ax1,ax2]
        for ax in axlist2:
            ax.set_ylabel('Height [km]',fontsize=Fontsize)
            ax.set_ylim(0,y_height)
                   
        ax1.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
        ax1a.set_xlabel('q [g kg$^{-1}$]',fontsize=Fontsize)
        ax2.set_xlabel('RH [%]',fontsize=Fontsize)
        #ax3.set_xlabel('Wind Speed [m s$^{-1}$]',fontsize=Fontsize)
 
        #------------------------------
        # plot temperature, q, & theta
        #------------------------------
        dumid = np.where(sonde_height[jj] <= y_height)
        tmp_height = sonde_height[jj][dumid]
        tmp_temp = sonde_temperature[jj][dumid]-273.15
        tmp_q = sonde_q[jj][dumid]
        tmp_theta = sonde_theta[jj][dumid]
        ax1.plot(tmp_temp,tmp_height,lw=2,c='red',ls='solid')
        ax1a.plot(tmp_q,tmp_height,lw=2,c='green',ls='solid')
        
        # deal with axis colors
        ax1.spines['bottom'].set_color('red')
        ax1a.spines['top'].set_color('green')
        ax1.xaxis.label.set_color('red')
        ax1a.xaxis.label.set_color('green')
        ax1.tick_params(axis='x', colors='red')
        ax1a.tick_params(axis='x', labelsize=Fontsize, colors='green')
        ax1a.spines['bottom'].set_visible(False)        

        # add theta to plot
        ax1b = ax1.twiny()
        ax1b.plot(tmp_theta,tmp_height,lw=2,c='darkorange')
        ax1b.set_xlabel('$\\theta$ [K]',fontsize=Fontsize)
        #theta_max = sonde_theta[jj][y_height_id]+3.
        #theta_min = sonde_theta[jj][0]-3.
        #ax1b.set_xlim(theta_min,theta_max)
        ax1b.xaxis.set_label_position("bottom")
        ax1b.xaxis.set_ticks_position("bottom")
        ax1b.spines['bottom'].set_position(('axes', -0.3))
        ax1b.xaxis.label.set_color('darkorange')
        ax1b.tick_params(axis='x',labelsize=Fontsize,colors='darkorange')
        ax1b.spines['bottom'].set_color('darkorange')
        ax1b.spines['top'].set_color('green')

        # add zero degree isotherm
        zero_temp,zero_height_id = find_nearest(tmp_temp,0)
        ax1.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon',label='0$^{\\circ}$C')
        ax1.legend(fontsize=Fontsize*0.9,loc='lower center',bbox_to_anchor=(0.375,0.05),framealpha=1,facecolor='white',edgecolor='black')
        
        ax1a.grid(False)
        ax1b.grid(False)
        
        #------------------------------
        # plot RH
        #------------------------------
        dumid = np.where(sonde_height[jj] <= y_height)
        tmp_height = sonde_height[jj][dumid]
        tmp_rh = sonde_rh[jj][dumid]
        tmp_rh_i = sonde_rh_i[jj][dumid]
        ax2.plot(tmp_rh,tmp_height,lw=2,c='blue',ls='solid',label='RH$_{liq}$')
        ax2.plot(tmp_rh_i,tmp_height,lw=2,c='deepskyblue',ls='solid',label='RH$_{ice}$')
        ax2.set_xlim(0,105.)
        ax2.axvline(95.,c='black',ls='dashed',lw=2)
        
        lgnd99=ax2.legend(fontsize=Fontsize,loc='upper center',bbox_to_anchor=(0.5,1.45),ncol=1,framealpha=0)
        
        
        if num_cloud_objects > 0.:
            dumx=ax2.get_xlim()
            dumx = np.arange(dumx[0],dumx[1]+1)
            for kk in range(num_cloud_objects):
                ax2.fill_between(dumx,sonde_cbhs[kk]*1.e-3,sonde_cths[kk]*1.e-3,facecolor='purple',alpha=0.25)
                ax2.axhline(sonde_cbhs[kk]*1.e-3,lw=2,c='purple')
                ax2.axhline(sonde_cths[kk]*1.e-3,lw=2,c='purple')

        
        #black_patch = mpatches.Patch(color='purple',alpha=0.25,label='RH Cloud Layer')
        #legend_elements2 = [Line2D([0], [0],color='purple',ls='solid',lw=3,label='RH CTH/CBH')]

        #lgndxyz = ax2.legend(handles=[black_patch],\
        #                    fontsize=Fontsize,\
        #                    bbox_to_anchor=(1,0.63),\
        #                    ncol=2,loc='upper left') 
        ##lgndxyy = ax2.legend(handles=legend_elements2,\
        #                    fontsize=Fontsize,\
        #                    bbox_to_anchor=(1,0.45),\
        #                    ncol=2,loc='upper left')         

        
        legend_elements = [Line2D([0], [0],color='black',ls='dashed',lw=2,label='95% RH')]
        lgnd97=ax2.legend(handles=legend_elements, loc='center left',\
               ncol=1,fontsize=Fontsize*0.9,framealpha=1,facecolor='white',edgecolor='black') 
        
        
        ax2.add_artist(lgnd99)
        #ax2.add_artist(lgndxyz)
        #ax2.add_artist(lgndxyy)
        ax2.add_artist(lgnd97)
        
        
        #------------------------------
        # satellite
        #------------------------------             
        if np.nanmax(sat_vis_before) > 0.:
            sat_var = sat_vis_before
            sat_levs = np.arange(0,1.05,0.05)
            sat_cmap = 'bone'
            sat_ylabel = 'Visible Reflectance'
            sat_ticks = [0,0.2,0.4,0.6,0.8,1.]
            dum_extend='neither'
        else:
            sat_var = sat_irtb_before
            sat_levs = np.arange(-25,5.5,0.5)
            sat_cmap = 'RdYlBu_r'
            sat_ylabel = '10.8 $\\mu$m T$_{b}$ [$^{\\circ}$C]'
            sat_ticks = [-25,-20,-15,-10,-5,0,5]
            dum_extend='both'
            
        # Current Time
        sat_plot = ax3.contourf(sat_lon,sat_lat,sat_var,cmap=sat_cmap,\
                                        levels=sat_levs,extend=dum_extend)
        ax3.set_xlabel('Longitude [$^{\\circ}$]',fontsize=Fontsize)
        ax3.set_ylabel('Latitude [$^{\\circ}$]',fontsize=Fontsize)
        ax3.tick_params(labelsize=Fontsize*0.85)
        cbar_sat_ax = fig.add_axes([0.92,0.75,0.02,0.13])
        cbar_sat = fig.colorbar(sat_plot,orientation='vertical',cax=cbar_sat_ax,\
                           ticks=sat_ticks,extend=dum_extend)
        cbar_sat.ax.tick_params(labelsize=Fontsize*0.85)
        cbar_sat.ax.set_ylabel(sat_ylabel,fontsize=Fontsize)
        tmp_time = sat_nearest_time.strftime('%m/%d/%Y %H:%M UTC')
        ax3.set_title(tmp_time,fontsize=Fontsize*1.15)
        mac_lat = -54.62
        mac_lon = 158.86
        ax3.plot(mac_lon,mac_lat,marker='*',c='magenta',markersize=15)
        legend_elements = [Line2D([0], [0], marker='*', color='w',label='Macquarie Island',
                  markerfacecolor='magenta', markersize=15)] 
        
        ax3.legend(handles=legend_elements,loc='lower center',\
                    bbox_to_anchor=(0.5,1.075),fontsize=Fontsize,framealpha=0)  

        #ax3.grid(which='both',axis='both',c='grey')

        #------------------------------
        # plot reflectivity & Doppler velocity
        #------------------------------         
        if basta_present_flag == True:

                
            dum_time_delta = datetime.timedelta(seconds=15)
            start_time_ts = toTimestamp(start_time)
            end_time_ts = toTimestamp(end_time)
            dumx = np.arange(start_time_ts,end_time_ts,1)
            dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])

            dum_diff = np.diff(basta_time_lim)
            dum_diff_0 = np.array([datetime.timedelta(seconds=0)])
            dum_diff = np.concatenate((dum_diff,dum_diff_0))
            uniq_diff = np.unique(dum_diff)
            dumid = np.where(dum_diff > datetime.timedelta(seconds=30))
            if np.size(dumid) > 0.:
                basta_ref_lim[:,dumid] = np.nan               
                basta_vel_lim[:,dumid] = np.nan               
                        
            
            
            basta_ref_lim[(basta_ref_lim > -999.) & (basta_ref_lim < -40.)] = -40.
            basta_ref_lim[basta_ref_lim > 10.] = 10.

            #from palettable.colorbrewer.sequential import GnBu_9_r as colormap
            #from palettable.cmocean.sequential import Ice_20 as colormap
            from palettable.cmocean.sequential import Solar_20 as colormap
            cmap = colormap.mpl_colormap
            cmap = plt.cm.cividis
            
            cmap.set_under('w')
            cmap.set_bad('grey')
            basta_bin_edges = np.arange(0,np.max(basta_height_lim)+12.5+25.,25.)
            ref_plot = ax4.pcolormesh(basta_time_lim,\
                                            basta_bin_edges/1.e3,\
                                            basta_ref_lim[:,:-1],\
                                            cmap=cmap,\
                                            vmin=-40.1,vmax=10.1,\
                                            shading='flat'
                                           )            
            
            
       

            basta_vel_lim[(basta_vel_lim > -999.) & (basta_vel_lim < -2.)] = -2.
            basta_vel_lim[basta_vel_lim > 2.] = 2.
            cmap = plt.cm.seismic
            cmap.set_under('w')
            cmap.set_bad('grey')
            vel_plot = ax5.pcolormesh(basta_time_lim,\
                                            basta_bin_edges/1.e3,\
                                            basta_vel_lim[:,:-1],\
                                            cmap=cmap,\
                                            vmin=-2.1,vmax=2.1,\
                                            shading='flat'
                                           )            

            
            if ~np.isnan(np.nanmax(merged_ceil_cbh_1_lim)):
                cbh_1_plot = ax4.scatter(basta_time_lim,merged_ceil_cbh_1_lim*1.e-3,\
                    s=5,marker='o',color='black',label='CBH 1')
                cbh_1_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_1_lim*1.e-3,\
                    s=5,marker='o',color='black',label='CBH 1')
            if ~np.isnan(np.nanmax(merged_ceil_cbh_2_lim)):
                cbh_2_plot = ax4.scatter(basta_time_lim,merged_ceil_cbh_2_lim*1.e-3,\
                    s=5,marker='o',color='magenta',label='CBH 2')
                cbh_2_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_2_lim*1.e-3,\
                    s=5,marker='o',color='magenta',label='CBH 2')
            if ~np.isnan(np.nanmax(merged_ceil_cbh_3_lim)):
                cbh_3_plot = ax4.scatter(basta_time_lim,merged_ceil_cbh_3_lim*1.e-3,\
                    s=5,marker='o',color='aqua',label='CBH 3')                  
                cbh_3_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_3_lim*1.e-3,\
                    s=5,marker='o',color='aqua',label='CBH 3')  
                
            ax4.xaxis.set_major_formatter(dfmt)
            ax5.xaxis.set_major_formatter(dfmt)
            ax4.set_xlim(start_time,end_time)
            ax5.set_xlim(start_time,end_time)
            ax4.set_ylabel('Height [km]',fontsize=Fontsize)
            ax4.set_xlabel('UTC Time [HH:MM]',fontsize=Fontsize)
            ax5.set_xlabel('UTC Time [HH:MM]',fontsize=Fontsize)
            ax5.set_ylabel('Height [km]',fontsize=Fontsize)
            ax4.set_ylim(0,y_height)
            ax5.set_ylim(0,y_height)
            ax4.plot(sonde_time_dt_long[jj],sonde_height[jj],lw=3,c='k',ls='solid')
            ax5.plot(sonde_time_dt_long[jj],sonde_height[jj],lw=3,c='k',ls='solid')
            #ax4.grid(which='both',c='grey',axis='both')
            #ax5.grid(which='both',c='grey',axis='both')

            # new ax with dimensions of the colorbar
            cbar_ax1 = fig.add_axes([0.92,0.535,0.02,0.135])
            dum_ticks = [-50,-40,-30,-20,-10,0,10,20]
            cbar1 = fig.colorbar(ref_plot, cax=cbar_ax1,ticks=dum_ticks)       
            cbar1.ax.set_ylabel('Reflectivity [dBZ]',fontsize=Fontsize)
            cbar1.ax.tick_params(labelsize=Fontsize)  
            
            # new ax with dimensions of the colorbar
            cbar_ax2 = fig.add_axes([0.92,0.3225,0.02,0.135])
            dum_ticks = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]
            cbar2 = fig.colorbar(vel_plot, cax=cbar_ax2,ticks=dum_ticks)       
            cbar2.ax.set_ylabel('Doppler Velocity [m s$^{-1}$]',fontsize=Fontsize)
            cbar2.ax.tick_params(labelsize=Fontsize)
            
 
            dumx=ax4.get_xlim()
            dumx = np.arange(dumx[0],dumx[1]+1)
            for kk in range(num_cloud_objects):
                ax4.fill_between(dumx,sonde_cbhs[kk]*1.e-3,\
                                 sonde_cths[kk]*1.e-3,facecolor='purple',alpha=0.25)
                ax4.axhline(sonde_cbhs[kk]*1.e-3,lw=2,c='purple')
                ax4.axhline(sonde_cths[kk]*1.e-3,lw=2,c='purple')
                ax5.fill_between(dumx,sonde_cbhs[kk]*1.e-3,\
                                 sonde_cths[kk]*1.e-3,facecolor='purple',alpha=0.25)
                ax5.axhline(sonde_cbhs[kk]*1.e-3,lw=2,c='purple')
                ax5.axhline(sonde_cths[kk]*1.e-3,lw=2,c='purple')
                

                
            dum_time_delta = datetime.timedelta(seconds=60)
            start_time_ts = toTimestamp(start_time)
            end_time_ts = toTimestamp(end_time)
            dumx = np.arange(start_time_ts,end_time_ts,1)
            dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])

            dumid = np.where(dumx_dt > basta_time_lim[-1])
            if np.size(dumid) > 0.:
                ax4.fill_between(dumx_dt[dumid],0,max_height,facecolor='grey')
                ax5.fill_between(dumx_dt[dumid],0,max_height,facecolor='grey')
                
            dumid = np.where(dumx_dt < basta_time_lim[0])
            if np.size(dumid) > 0.:
                ax4.fill_between(dumx_dt[dumid],0,max_height,facecolor='grey')
                ax5.fill_between(dumx_dt[dumid],0,max_height,facecolor='grey')
                               
                
            black_patch = mpatches.Patch(color='purple',alpha=0.25,label='RH Cloud Layer')
            legend_elements2 = [Line2D([0], [0],color='purple',ls='solid',lw=3,label='RH CTH/CBH')]

            lgnd999 = ax4.legend(handles=[black_patch],\
                                fontsize=Fontsize,\
                                bbox_to_anchor=(0.5,1.35),\
                                ncol=2,loc='upper center',framealpha=0) 
            lgnd998 = ax4.legend(handles=legend_elements2,\
                                fontsize=Fontsize,\
                                bbox_to_anchor=(0.5,1.235),\
                                ncol=2,loc='upper center',framealpha=0) 
            
            grey_patch = mpatches.Patch(color='grey',label='Invalid Radar Data')
            
            lgnd = ax5.legend(handles=[grey_patch],\
                                fontsize=Fontsize,\
                                bbox_to_anchor=(0.56,1.275),\
                                ncol=1,loc='upper center',framealpha=0)            

            legend_elements = [Line2D([0], [0], marker='.', color='w', label='CBH',
                          markerfacecolor='black', markersize=15)]
            lgnd3=ax5.legend(handles=legend_elements, loc='upper left',\
                   ncol=3,fontsize=Fontsize,bbox_to_anchor=(0.675,1.275),framealpha=0)             

            legend_elements = [Line2D([0], [0],color='black',ls='solid',lw=3,label='Sonde Path')]
            lgnd4=ax5.legend(handles=legend_elements, loc='upper left',\
                   ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.17,1.275),framealpha=0)  
        
            
            # add zero degree isotherm
            zero_temp,zero_height_id = find_nearest(tmp_temp,0)
            ax4.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon',label='0$^{\\circ}$C')
            ax5.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon',label='0$^{\\circ}$C')
            legend_elements = [Line2D([0], [0],color='maroon',ls='dashed',lw=2,label='0 $^{\\circ}$C')]                                                  
            lgndxxx = ax5.legend(handles=legend_elements,fontsize=Fontsize,loc='upper left',\
                                 bbox_to_anchor=(0.025,1.275),framealpha=0,facecolor='white',edgecolor='black')
                    
            
            
            ax5.add_artist(lgnd4)
            ax5.add_artist(lgnd3)
            ax5.add_artist(lgnd)
            ax4.add_artist(lgnd999)
            ax4.add_artist(lgnd998)

           
        else:
            ax4.set_xlim(0,1)
            ax4.set_ylim(0,1)
            tmpplot=ax4.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
            ax4.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
            tmpplot[0].axes.get_xaxis().set_visible(False)
            tmpplot[0].axes.get_yaxis().set_visible(False)    
            ax4.set_title('No BASTA Data Available',fontsize=Fontsize*2.,c='red') 
            
            ax5.set_xlim(0,1)
            ax5.set_ylim(0,1)
            tmpplot=ax5.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
            ax5.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
            tmpplot[0].axes.get_xaxis().set_visible(False)
            tmpplot[0].axes.get_yaxis().set_visible(False)    
            ax5.set_title('No BASTA Data Available',fontsize=Fontsize*2.,c='red')  
 
        #-----------------------------------
        # Plot ARM Ceilometer
        #-----------------------------------
        if ceilometer_present == True:
 

            dumid = np.where((ceil_time_dt >= start_time) & (ceil_time_dt <= end_time))
            if np.size(dumid) == 0.:
                ax6.set_xlim(0,1)
                ax6.set_ylim(0,1)
                tmpplot=ax6.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
                ax6.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
                tmpplot[0].axes.get_xaxis().set_visible(False)
                tmpplot[0].axes.get_yaxis().set_visible(False)    
                ax6.set_title('No ARM Ceilometer Data Available',fontsize=Fontsize*2.,c='red')             
            else:
                
                ax6.set_ylabel('Height [km]',fontsize=Fontsize)
                ax6.set_ylim(0,y_height)
                ax6.set_xlabel('UTC Time [HH:MM]',fontsize=Fontsize)
                ax6.plot(sonde_time_dt_long[jj],sonde_height[jj],lw=3,c='k',ls='solid')
                ax6.xaxis.set_major_formatter(dfmt)
                ax6.set_xlim(start_time,end_time)                  
                dumid = np.squeeze(dumid)
                ceil_cbh_1_native_lim = ceil_cbh_1_native[dumid]
                ceil_cbh_2_native_lim = ceil_cbh_2_native[dumid]
                ceil_cbh_3_native_lim = ceil_cbh_3_native[dumid]
                ceil_backscatter_lim = ceil_backscatter[dumid,:]
                
                ceil_backscatter_lim = np.squeeze(ceil_backscatter_lim)
                ceil_time_dt_lim = ceil_time_dt[dumid]
                tmpid = np.where(ceil_range <= y_height*1.e3)
                tmpid = np.squeeze(tmpid)
                #ceil_backscatter_lim = ceil_backscatter_lim[:,tmpid]
                ceil_backscatter_lim = ceil_backscatter_lim[:,:]
                #ceil_range_lim = ceil_range[tmpid]
                ceil_range_lim = ceil_range[:]
                ceil_backscatter_lim = ceil_backscatter_lim*1.e-3/10000.  
                #print(aaaaa)
                dumid = np.where((~np.isnan(ceil_backscatter_lim)) & (ceil_backscatter_lim > 0.) )
                dum = ceil_backscatter_lim.copy()

                dum[dumid] = np.log10(dum[dumid])
                dumid = np.where(ceil_backscatter_lim == 0.)
                dum[dumid] = np.nan
                dumid = np.where(ceil_backscatter_lim < 0.)
                dum[dumid] = np.nan
                dum[np.isnan(dum)] = 0.

                
                dum_time_delta = datetime.timedelta(seconds=60)
                start_time_ts = toTimestamp(start_time)
                end_time_ts = toTimestamp(end_time)
                dumx = np.arange(start_time_ts,end_time_ts,1)
                dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])
                
                
                dum_diff = np.diff(ceil_time_dt_lim)
                dum_diff_0 = np.array([datetime.timedelta(seconds=0)])
                dum_diff = np.concatenate((dum_diff,dum_diff_0))
                #uniq_diff = np.unique(dum_diff)
                dumid = np.where(dum_diff > datetime.timedelta(seconds=30))
                if np.size(dumid) > 0.:
                    dum[dumid] = np.nan          
                
                
                
                #cmap = plt.cm.jet
                from palettable.cmocean.sequential import Haline_20 as colormap
                #from palettable.cmocean.sequential import Thermal_20 as colormap
                cmap = colormap.mpl_colormap
                cmap = plt.cm.viridis
                #cmap = plt.cm.cividis
                cmap.set_under('black')
                cmap.set_bad('grey')
                dum[dum == 0.] = np.nan
                dum[dum < -8.] = -8.
                dum[dum > -3.] = -3.

                ceil_backscatter_plot = ax6.pcolormesh(ceil_time_dt_lim,\
                                                     ceil_range_lim*1.e-3,\
                                                     np.transpose(dum),\
                                                     cmap=cmap,\
                                                     vmin=-8,vmax=-3)

                if ~np.isnan(np.nanmax(ceil_cbh_1_native_lim)):
                    cbh_1_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_1_native_lim*1.e-3,\
                        s=5,marker='o',color='black',label='CBH 1')
                #if ~np.isnan(np.nanmax(ceil_cbh_2_native_lim)):
                #    cbh_2_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_2_native_lim*1.e-3,\
                #        s=5,marker='o',color='magenta',label='CBH 2')
                #if ~np.isnan(np.nanmax(ceil_cbh_3_native_lim)):
                #    cbh_3_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_3_native_lim*1.e-3,\
                #        s=5,marker='o',color='aqua',label='CBH 3')                    
                    
                cbar_ax3 = fig.add_axes([0.92,0.11,0.02,0.135])
                # new ax with dimensions of the colorbar
                dum_ticks = [-8,-7,-6,-5,-4,-3]
                cbar3 = fig.colorbar(ceil_backscatter_plot, cax=cbar_ax3,ticks=dum_ticks)       
                cbar3.ax.set_ylabel('$log_{10}$($\\beta_{att}$) [sr$^{-1}$ m$^{-1}$]',fontsize=Fontsize)
                cbar3.ax.tick_params(labelsize=Fontsize)
                #ax6.grid(which='both',c='grey',axis='both')
                ax6.set_title('ARM Ceilometer',fontsize=Fontsize*1.5)
                
                
                dum_time_delta = datetime.timedelta(seconds=60)
                start_time_ts = toTimestamp(start_time)
                end_time_ts = toTimestamp(end_time)
                dumx = np.arange(start_time_ts,end_time_ts,1)
                dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])

                dumid = np.where(dumx_dt > ceil_time_dt_lim[-1])
                if np.size(dumid) > 0.:
                    ax6.fill_between(dumx_dt[dumid],0,y_height,facecolor='grey')

                dumid = np.where(dumx_dt < ceil_time_dt_lim[0])
                if np.size(dumid) > 0.:
                    ax6.fill_between(dumx_dt[dumid],0,y_height,facecolor='grey')
                
                
                
                
                legend_elements = [Line2D([0], [0], marker='.', color='w', label='CBH',
                              markerfacecolor='black', markersize=15)]
                lgnd2=ax6.legend(handles=legend_elements, loc='upper left',\
                       ncol=3,fontsize=Fontsize,bbox_to_anchor=(0.75,1.225),framealpha=0)
                   
  
                legend_elements = [Line2D([0], [0],color='black',ls='solid',lw=3,label='Sonde Path')]
                lgnd5=ax6.legend(handles=legend_elements, loc='upper left',\
                       ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.05,1.225),framealpha=0)  


                ax6.add_artist(lgnd2)
                ax6.add_artist(lgnd5)
                
                
        else:
            ax6.set_xlim(0,1)
            ax6.set_ylim(0,1)
            tmpplot=ax6.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
            ax6.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
            tmpplot[0].axes.get_xaxis().set_visible(False)
            tmpplot[0].axes.get_yaxis().set_visible(False)    
            ax6.set_title('No ARM Ceilometer Data Available',fontsize=Fontsize*2.,c='red') 
        
            
            
        plt.subplots_adjust(wspace=0.375,hspace=0.6)
        
        # Plot Date
        tmp_time = current_date_sonde_times_dt[jj].strftime("%m/%d/%Y %H%M")
        plt.figtext(0.5,0.94,'Sounding Release Time: '+tmp_time+' UTC',\
                    fontsize=Fontsize*1.8,ha='center')
        
        ax1.text(-0.35,1.2,'(a)',fontsize=Fontsize*2.5,transform=ax1.transAxes)
        ax2.text(-0.35,1.2,'(b)',fontsize=Fontsize*2.5,transform=ax2.transAxes)
        ax2.text(-0.35,1.2,'(c)',fontsize=Fontsize*2.5,transform=ax3.transAxes)
        ax4.text(-0.11,1.1,'(d)',fontsize=Fontsize*2.5,transform=ax4.transAxes)
        ax5.text(-0.11,1.1,'(e)',fontsize=Fontsize*2.5,transform=ax5.transAxes)
        ax6.text(-0.11,1.1,'(f)',fontsize=Fontsize*2.5,transform=ax6.transAxes)
        
        ax1.yaxis.set_ticks_position("left")        
        ax2.xaxis.set_ticks_position("bottom")
        ax2.yaxis.set_ticks_position("left")        
        ax3.xaxis.set_ticks_position("bottom")
        ax3.yaxis.set_ticks_position("left")
        ax4.xaxis.set_ticks_position("bottom")
        ax4.yaxis.set_ticks_position("left")
        ax5.xaxis.set_ticks_position("bottom")
        ax5.yaxis.set_ticks_position("left")        
        ax6.xaxis.set_ticks_position("bottom")
        ax6.yaxis.set_ticks_position("left")
        
        ax1.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)
        ax2.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)
        ax3.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)
        ax4.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)
        ax5.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)  
        ax6.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=2)
        ax3.set_axisbelow(False)
        ax4.set_axisbelow(False)
        ax5.set_axisbelow(False)
        ax6.set_axisbelow(False)
        
        #plt.show()
        #plt.close()
        #print(aaaaa)
        
        fig_path = '/home/mwstanfo/figures/micre_paper/'
        dum_time = current_date_sonde_times_dt[jj].strftime('%Y%m%d_%H%M')
        #outfile = 'sounding_summary_'+dum_time+'UTC.png'
        outfile = 'fig_01.png'
        #outfile = 'fig_01.eps'
        plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
        plt.close()
    print(aaaaa)
        
        #print(aaaa)
    #print(aaaa)
    #if ii == 30:
    #    print(aaaaa)
    #===========================================
    # End Plot
    #===========================================
    
    ii+=1
    continue
    print(aaaa)    

    
