#=========================================================
# calc_rh_at_ceilometer_cbh.py
# Calculates the relative humidity at the ceilometer CBH
# using soundings.
# Produces Fig. D1 in the manuscript.
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



#--------------------------------------------------------------
# Use soundings to define timeline
# and exclude days without any soundings
#--------------------------------------------------------------
dumid = np.where((sonde_dates_dt >= basta_dates_dt[0]) & (sonde_dates_dt <= basta_dates_dt[-1]))
sonde_dates_dt = sonde_dates_dt[dumid]
sonde_times_dt = sonde_times_dt[dumid]
sonde_files = sonde_files[dumid]
dates = sonde_dates_dt.copy()
times = sonde_times_dt.copy()
unique_dates = np.unique(dates)


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

sonde_height_30_min_all = []
sonde_rh_cbh_all = []
sonde_time_all = []
sonde_temp_all = []
sonde_rh_all = []

ii = 0
ii_end = 15
for date in unique_dates[ii:]:
    
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
    sonde_rh = []
    sonde_rh_i = []
    sonde_height = []
    sonde_time_dt_long = []
    sonde_time_dt = []

    jj_ind = []  
    for jj in range(num_current_soundings):
        
        current_date_sonde_file = current_date_sonde_files[jj]
        current_date_sonde_file = current_date_sonde_file.split('/')[-1]
        path = '/mnt/raid/mwstanfo/micre_data/micre_soundings/'

        file_size = os.stat(path+current_date_sonde_file).st_size/1.e3
        fstruct = fs(current_date_sonde_file,path,file_size)
        Sondetmp = load_sonde_data(fstruct)
        
        max_alt = np.max(Sondetmp['alt'])

        if max_alt < 8.:
            print('Sonde failed to reach 10 km. Therefore omitting this sounding.')
            #np.delete(current_date_sonde_times_dt,jj)
            #np.delete(current_date_sonde_files,jj)
            jj_ind.append(jj)
            #continue
        # Skip soundings where RH specifically doesn't reach 10 km
        tmpid = np.where(np.isnan(Sondetmp['RH']))
        if np.size(tmpid) > 0:
            dumid = np.where(~np.isnan(Sondetmp['RH']))
            dum_alt = Sondetmp['alt'][dumid]
            if np.size(dumid) > 1:
                max_alt_rh = dum_alt[-1]
            else:
                max_alt_rh = dum_alt[0]
            if max_alt_rh < 8.:
                jj_ind.append(jj)
                #continue
            
        sonde_temperature.append(Sondetmp['drybulb_temp'])
        sonde_rh.append(Sondetmp['RH'])
        sonde_height.append(Sondetmp['alt'])
        Moretmp = calculate_theta_and_more(Sondetmp['drybulb_temp'],Sondetmp['pressure'],\
                                           RH=Sondetmp['RH'],use_T_K=True,\
                                          sat_pres_formula='Emmanuel')
        sonde_rh_i.append(Moretmp['RH_i'])        
        sonde_time_dt_long.append(Sondetmp['time'])
        sonde_time_dt.append(current_date_sonde_times_dt[jj])
        
    if np.size(jj_ind) > 0.:
        for jj in range(len(jj_ind)):    
            current_date_sonde_files = np.delete(current_date_sonde_files,jj_ind[jj])
            current_date_sonde_times_dt = np.delete(current_date_sonde_times_dt,jj_ind[jj])
            sonde_id = np.delete(sonde_id,jj_ind[jj])
            sonde_temperature = np.delete(sonde_temperature,jj_ind[jj])
            sonde_rh = np.delete(sonde_rh,jj_ind[jj])
            sonde_height = np.delete(sonde_height,jj_ind[jj])
            sonde_rh_i = np.delete(sonde_rh_i,jj_ind[jj])
            sonde_time_dt = np.delete(sonde_time_dt,jj_ind[jj])
            sonde_time_dt_long = np.delete(sonde_time_dt_long,jj_ind[jj])
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
    
    # Get a "target_start_time" and "target_end_time" which is the sounding release time
    # and 15 minutes after the release time, respectively. Also store the
    # the height that the sonde reaches within 15 minutes.
    for jj in range(num_current_soundings):
        current_sonde_time_dt_long = sonde_time_dt_long[jj]
        current_sonde_time_dt_long = np.array(current_sonde_time_dt_long)
        current_sonde_height = sonde_height[jj]
        current_sonde_height = np.array(current_sonde_height)
        target_start_time = current_sonde_time_dt_long[0]      
        target_end_time = current_sonde_time_dt_long[0]+datetime.timedelta(minutes=30)
        dumid = np.where(current_sonde_time_dt_long <= target_end_time)
        dumid = np.squeeze(dumid)
        height_at_30_min = current_sonde_height[dumid[-1]]
        sonde_height_30_min_all.append(height_at_30_min)

    #===========================================
    # End sounding block
    #===========================================
    print('Completed sounding processing.')

    #ii+=1


    
    #======================================================
    # Begin radar block
    #======================================================
    print('Begin radar processing.')
    dumid = np.where(basta_dates_dt == date)
    if np.size(dumid) == 0.:
        print('No radar data on this date. Ommitting sounding.')
        basta_present_flag = False
        #ii+=1
        #continue
    else:
        basta_present_flag = True
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
        # we pull in at least 30 minutes past the final sounding.
        #------------------------------------------------------
        target_start_time = current_date_sonde_times_dt[0]
        target_end_time = current_date_sonde_times_dt[-1]+datetime.timedelta(minutes=30)
        #target_end_time = target_start_time+datetime.timedelta(minutes=30)

        if ii != (len(basta_dates_dt)-1):
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


        basta_date_dt = np.array([datetime.datetime(basta_time_dt[dd].year,\
                                                    basta_time_dt[dd].month,\
                                                    basta_time_dt[dd].day) for dd in range(len(basta_time_dt))])

    # Check to see if any of the dates in the current file equal the before date
        tmpid = np.where((basta_time_dt >= target_start_time) & (basta_time_dt <= target_end_time))
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            # limit arrays
            basta_time_dt = basta_time_dt[tmpid]
            basta_flag_coupling = basta_flag_coupling[tmpid]
            basta_flag = basta_flag[:,tmpid]
            basta_ref = basta_ref[:,tmpid]
            basta_vel = basta_vel[:,tmpid]
            basta_noise_level = basta_noise_level[tmpid]

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
        # We will also assing all basta_flag values up to 137.5 m
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
        #raise RuntimeError('No ARM files at all for this date.')
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
                
        
        #ceil_cbh_1_native = ceil_cbh_1.copy()
        #ceil_cbh_2_native = ceil_cbh_2.copy()
        #ceil_cbh_3_native = ceil_cbh_3.copy()
        
        
        # Limit to the start time of the first sounding
        # and 30 minutes beyond the end time of the second
        # sounding
        dumid = np.where( (ceil_time_dt >= target_start_time) & (ceil_time_dt <= target_end_time))
        if np.size(dumid) > 0.:
            dumid = np.squeeze(dumid)
            ceil_time_dt = ceil_time_dt[dumid]
            ceil_time_ts = ceil_time_ts[dumid]
            ceil_cbh_1 = ceil_cbh_1[dumid]
            ceil_qc_cbh_1 = ceil_qc_cbh_1[dumid]
            ceil_cbh_2 = ceil_cbh_2[dumid]
            ceil_qc_cbh_2 = ceil_qc_cbh_2[dumid]
            ceil_cbh_3 = ceil_cbh_3[dumid]
            ceil_qc_cbh_3 = ceil_qc_cbh_3[dumid]
            ceil_status_flag = ceil_status_flag[dumid]
            ceil_detection_status = ceil_detection_status[dumid]
            ceil_backscatter = ceil_backscatter[dumid,:]

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
        #if np.size(tmpid) > 0.:
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
        #ceil_cbh_1 = ceil_cbh_1_interp.copy()
        #ceil_detection_status = ceil_detection_status_interp.copy()
        #del ceil_cbh_1_interp,ceil_detection_status_interp
    
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
        print('No AAD ceilometer files for this date.')
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

                #dumiid = np.where(after_aad_ceil_time_dt <= target_end_time)

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

            # Limit to the start time of the first sounding
            # and 30 minutes beyond the end time of the second
            # sounding
            dumid = np.where( (aad_ceil_time_dt >= target_start_time) & (aad_ceil_time_dt <= target_end_time))
            if np.size(dumid) > 0.:
                dumid = np.squeeze(dumid)
                aad_ceil_time_dt = aad_ceil_time_dt[dumid]
                aad_ceil_cbh_1 = aad_ceil_cbh_1[dumid]
                aad_ceil_cbh_2 = aad_ceil_cbh_2[dumid]
                aad_ceil_cbh_3 = aad_ceil_cbh_3[dumid]
                aad_ceil_detection_status = aad_ceil_detection_status[dumid]
                aad_ceil_backscatter = aad_ceil_backscatter[dumid,:]            
            
            
            
            #------------------------------------------
            # Interpolate ceilometer to radar time grid
            # using nearest neighbor interpolation. 
            # This method requires that the nearest
            # neighbor be within 15 seconds of the
            # radar time grid element.
            #------------------------------------------
            basta_time_ts = np.array([toTimestamp(basta_time_dt[dd]) for dd in range(len(basta_time_dt))])
            #if np.size(tmpid) > 0.:
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

                    #if np.isnan(nearest_aad_ceil_detection_status):
                    #    aad_ceil_cbh_1_interp.append(np.nan)
                    #    continue

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
            #ceil_cbh_1 = ceil_cbh_1_interp.copy()
            #ceil_detection_status = ceil_detection_status_interp.copy()
            
            if False:
                fig = plt.figure(figsize=(8,5))
                ax1 = fig.add_subplot(111)
                #ax1.scatter(aad_ceil_time_dt,aad_ceil_cbh_1,s=20,marker='.',c='blue')
                #ax1.scatter(basta_time_dt,aad_ceil_cbh_1_interp,s=10,marker='.',c='red')
                ax1.scatter(basta_time_dt,aad_ceil_detection_status_interp,s=10,marker='.',c='red')
                plt.show()
            
            
            print('End AAD ceilometer processing.')    
                        
        else:
            print('No AAD ceilometer files for this date.')
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
    sonde_rh_interp = []
    sonde_rh_i_interp = []
    for jj in range(len(sonde_time_dt)):
        sonde_temperature_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_temperature[jj]))
        sonde_rh_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_rh[jj]))
        sonde_rh_i_interp.append(np.interp(basta_height,sonde_height[jj]*1.e3,sonde_rh_i[jj]))
            

    sonde_temperature_interp = np.array(sonde_temperature_interp)
    sonde_rh_interp = np.array(sonde_rh_interp)
    sonde_rh_i_interp = np.array(sonde_rh_i_interp)

    print('Completed sounding interpolation.')    
    #===========================================
    # End Sonde Interpolation
    #===========================================  
    
    #===========================================
    # Start RH @ CBH
    #===========================================
    for jj in range(num_current_soundings):
        start_time = sonde_time_dt[jj]
        end_time = sonde_time_dt[jj] + datetime.timedelta(minutes=20)

        # grab current sonde variables
        current_sonde_time_dt_long = np.array(sonde_time_dt_long[jj])
        current_sonde_height = np.array(sonde_height[jj])
        current_sonde_rh= np.array(sonde_rh[jj])
        current_sonde_temperature= np.array(sonde_temperature[jj])
        current_sonde_rh_interp = np.array(sonde_rh_interp[jj])

        # limit basta times to those within 30 minutes of sounding release
        dumid = np.where((basta_time_dt >= start_time) & (basta_time_dt <= end_time))
        if np.size(dumid) > 0.:
            dumid = np.squeeze(dumid)
            # limit interpoalted variables to within 30 minutes of sounding release
            merged_ceil_cbh_1_lim = merged_ceil_cbh_1[dumid]
            if np.all(np.isnan(merged_ceil_cbh_1_lim)):
                continue
            basta_ref_lim = basta_ref[:,dumid]
            basta_time_dt_lim = basta_time_dt[dumid]
            basta_time_ts_lim = np.array([toTimestamp(basta_time_dt_lim[dd]) for dd in range(len(basta_time_dt_lim))])
            #current_sonde_rh_interp_lim = current_sonde_rh_interp[dumid]
            
            dumid2 = np.where(current_sonde_time_dt_long <= end_time)
            if np.size(dumid2) > 0.:
                # limit sonde long time and height, which are not interpoalted, to within 30 min. of sonde release
                current_sonde_time_dt_long_lim = current_sonde_time_dt_long[dumid2]
                current_sonde_height_lim = current_sonde_height[dumid2]
                current_sonde_rh_lim = current_sonde_rh[dumid2]
                current_sonde_temperature_lim = current_sonde_temperature[dumid2]
                #plt.hist(current_sonde_rh_lim)
                #plt.show()
                # Now interpolate the sonde height from the "long" time to the basta time
                current_sonde_time_ts_long_lim = np.array([toTimestamp(current_sonde_time_dt_long_lim[dd]) for dd in range(len(current_sonde_time_dt_long_lim))])
                current_sonde_height_lim_interp = np.interp(basta_time_ts_lim,current_sonde_time_ts_long_lim,current_sonde_height_lim)
                current_sonde_rh_lim_interp = np.interp(basta_time_ts_lim,current_sonde_time_ts_long_lim,current_sonde_rh_lim)
                current_sonde_temperature_lim_interp = np.interp(basta_time_ts_lim,current_sonde_time_ts_long_lim,current_sonde_temperature_lim)
                current_sonde_height_lim_interp = current_sonde_height_lim_interp*1.e3
                
                # Now that the sonde height is on the basta time grid, need
                # to also interpolate the height to the basta height grid
                # We'll do this like we did for the ceilometers, ensuring that the
                # height is within the basta bin edges
                
                # The basta height sometimes is brought out to 4 decimal places,
                # so need to round to the nearest tenth of a decimal place
                basta_height = np.around(basta_height,1)
                
                
                
                sonde_single_height_arr = []
                #rh_arr = []
                
                # Loop through the sonde heights interpolated to the basta height grid
                # and interpolated to basta midbins
                for kk in range(len(current_sonde_height_lim_interp)):
                    sonde_single_height = current_sonde_height_lim_interp[kk]
                    nearest_val,nearest_id = find_nearest(basta_height,sonde_single_height)
                    if nearest_id == 0.:
                        midbin = (basta_bin_edges[0]+basta_bin_edges[1])/2.
                        sonde_single_height_arr.append(midbin)
                    elif sonde_single_height == nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        dumid = np.where(basta_height == midbin)
                        sonde_single_height_arr.append(midbin)
                    elif sonde_single_height < nearest_val:
                        bin_edges = basta_bin_edges[nearest_id-1:nearest_id+1]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        dumid = np.where(basta_height == midbin)
                        sonde_single_height_arr.append(midbin)
                    elif sonde_single_height > nearest_val:
                        bin_edges = basta_bin_edges[nearest_id:nearest_id+2]
                        midbin = (bin_edges[0]+bin_edges[1])/2.
                        dumid = np.where(basta_height == midbin)
                        sonde_single_height_arr.append(midbin)

                                    
                sonde_height_interp_final = np.array(sonde_single_height_arr)

                
                
                # base height of RH after the height closest to the mean CBH
                if False:
                    #if True:
                    mean_cbh = np.nanmean(merged_ceil_cbh_1_lim)
                    nearest_val,nearest_id = find_nearest(sonde_height_interp_final,mean_cbh)
                    rh_at_cbh = current_sonde_rh_lim_interp[nearest_id]
                    sonde_cbh = nearest_val
                    temp_at_cbh = current_sonde_temperature_lim_interp[nearest_id]
                
                # produce histogram of CBHs
                #if False:
                if True:
                    bins = np.arange(0,8250,250)
                    hist,hist_bins = np.histogram(merged_ceil_cbh_1_lim,bins=bins)
                    hist_tot = np.sum(hist)
                    hist_norm = hist/hist_tot

                    if np.max(hist_norm) > 0.5:
                        dum_max = np.argmax(hist_norm)
                        dumid = np.where((merged_ceil_cbh_1_lim >= hist_bins[dum_max]) & (merged_ceil_cbh_1_lim < hist_bins[dum_max+1]) )
                        dum_cbh = merged_ceil_cbh_1_lim[dumid]
                        dum_rh = current_sonde_rh_lim_interp[dumid]
                        dum_height = sonde_height_interp_final[dumid]
                        mean_cbh = np.nanmean(dum_cbh)
                        nearest_val,nearest_id = find_nearest(sonde_height_interp_final,mean_cbh)
                        rh_at_cbh = current_sonde_rh_lim_interp[nearest_id]
                        temp_at_cbh = current_sonde_temperature_lim_interp[nearest_id]
                        sonde_cbh = nearest_val
                    else:
                        dumid = np.where(hist_norm >= 0.25)
                        if np.size(dumid) > 1.:
                            dumid = np.squeeze(dumid)
                            dum_max = np.min(dumid)
                            dumid2 = np.where((merged_ceil_cbh_1_lim >= hist_bins[dum_max]) & (merged_ceil_cbh_1_lim < hist_bins[dum_max+1]) )
                            dum_cbh = merged_ceil_cbh_1_lim[dumid2]
                            dum_rh = current_sonde_rh_lim_interp[dumid2]
                            dum_height = sonde_height_interp_final[dumid2]
                            mean_cbh = np.nanmean(dum_cbh)
                            nearest_val,nearest_id = find_nearest(sonde_height_interp_final,mean_cbh)
                            rh_at_cbh = current_sonde_rh_lim_interp[nearest_id]
                            temp_at_cbh = current_sonde_temperature_lim_interp[nearest_id]
                            sonde_cbh = nearest_val                        


                        else:
                            dum_max = np.argmax(hist_norm)
                            dumid = np.where((merged_ceil_cbh_1_lim >= hist_bins[dum_max]) & (merged_ceil_cbh_1_lim < hist_bins[dum_max+1]) )
                            dum_cbh = merged_ceil_cbh_1_lim[dumid]
                            dum_rh = current_sonde_rh_lim_interp[dumid]
                            dum_height = sonde_height_interp_final[dumid]
                            mean_cbh = np.nanmean(dum_cbh)
                            nearest_val,nearest_id = find_nearest(sonde_height_interp_final,mean_cbh)
                            rh_at_cbh = current_sonde_rh_lim_interp[nearest_id]
                            temp_at_cbh = current_sonde_temperature_lim_interp[nearest_id]
                            sonde_cbh = nearest_val                
                
                # Now loop through times and stop when the height
                # of the CBH and the height of the sonde is equal
                #diff_arr = []
                #for kk in range(len(sonde_height_interp_final)):
                #    diff_arr.append(np.abs(sonde_height_interp_final[kk]-merged_ceil_cbh_1_lim[kk]))

                #diff_arr = np.array(diff_arr)
                #min_val_arg = np.nanargmin(diff_arr)
                #min_val = diff_arr[min_val_arg]
                #sonde_cbh = sonde_height_interp_final[min_val_arg]
                #rh_at_cbh = current_sonde_rh_lim_interp[min_val_arg]
                    
                # Now take that kk_ind and determine the altitude difference between
                # that one and the height below it
                #alt_diff_at = np.abs(sonde_height_interp_final[kk_ind]-merged_ceil_cbh_1_lim[kk_ind])
                #alt_diff_below = np.abs(sonde_height_interp_final[kk_ind-1]-merged_ceil_cbh_1_lim[kk_ind])
                #if alt_diff_at == alt_diff_below:
                #    kk_ind = kk_ind
                #elif alt_diff_at > alt_diff_below:
                #    kk_ind = kk_ind-1
                #elif alt_diff_at < alt_diff_below:
                #    kk_ind = kk_ind


                
                
                sonde_rh_cbh_all.append(sonde_cbh)
                sonde_time_all.append(start_time)
                sonde_temp_all.append(temp_at_cbh-273.15)
                sonde_rh_all.append(rh_at_cbh)
                                
   
               # if True:
                if False:
                    print('Plotting...')
                    sns.set_theme()
                    #sns.set_style('dark')
                    sns.set_style('ticks')
                    sns.set(rc={'axes.facecolor':'white','axes.edgecolor': 'black'})
                    sns.set_context('talk')        
                    dfmt = mdates.DateFormatter('%H:%M')

                    y_height = 3
                    Fontsize=14
                    fig = plt.figure(figsize=(12.3,12.3))  
                    gs=GridSpec(3,1) # 4 rows, 3 columns
                    ax1 = fig.add_subplot(gs[0,0])
                    #ax1a = ax1.twiny()
                    ax2 = fig.add_subplot(gs[1,0])  
                    ax3 = fig.add_subplot(gs[2,0])  

                    axlist = [ax1,ax2,ax3]
                    for ax in axlist:
                        ax.tick_params(labelsize=Fontsize)
                        ax.set_ylabel('Height [km]',fontsize=Fontsize)
                        ax.set_ylim(0,y_height)
                        ax.set_xlabel('UTC Time [HH:MM]',fontsize=Fontsize)
                        ax.xaxis.set_major_formatter(dfmt)
                        ax.set_xlim(basta_time_dt_lim[0],basta_time_dt_lim[-1])
                        
                    #-------------------------------                    
                    # Soundings
                    #-------------------------------
                   # ax1.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
                   # ax1a.set_xlabel('q [g kg$^{-1}$]',fontsize=Fontsize)
                   # ax2.set_xlabel('RH [%]',fontsize=Fontsize)                                 
                    axa = ax1.twiny()
                    axb = ax1.twiny()
                    
                    ax1.plot(basta_time_dt_lim,sonde_height_interp_final*1.e-3,\
                            lw=3,c='black',ls='solid',label='Sonde Height')                   
                    axa.plot(current_sonde_rh_lim_interp,sonde_height_interp_final*1.e-3,\
                           lw=3,c='blue',ls='solid',label='RH') 
                    axb.plot(current_sonde_temperature_lim_interp-273.15,sonde_height_interp_final*1.e-3,\
                           lw=3,c='red',ls='solid',label='Temperature') 
                    
                    
                    # deal with axis colors
                    axa.spines['top'].set_color('blue')
                    axa.xaxis.label.set_color('blue')
                    axa.tick_params(axis='x', labelsize=Fontsize, colors='blue')
                    axa.spines['bottom'].set_visible(False)
                    axa.set_xlabel('RH [%]',fontsize=Fontsize)
                    axb.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
                    axb.xaxis.set_label_position("bottom")
                    axb.xaxis.set_ticks_position("bottom")
                    axb.spines["bottom"].set_position(('axes',-0.35))
                    axb.xaxis.label.set_color('red')
                    axb.tick_params(axis='x',labelsize=Fontsize,colors='red')
                    axb.spines['bottom'].set_color('red')
                    axb.spines['top'].set_color('blue')

                    
                    # CBH scatter
                    ax1.scatter(basta_time_dt_lim,merged_ceil_cbh_1_lim*1.e-3,s=10,\
                               color='magenta',label='CBH')
                    
                    axa.axhline(sonde_cbh*1.e-3,\
                              lw=3,c='green',ls='dashed',label='Sonde Loc. @ CBH')
                    axa.axvline(rh_at_cbh,\
                              lw=3,c='deepskyblue',ls='dashed',label='RH @ CBH') 
                    dumstr = 'RH @ CBH: '+str(np.around(rh_at_cbh,1))+'%'
                    axa.text(80,-0.5,dumstr,fontsize=Fontsize,\
                            fontweight='bold',c='deepskyblue',ha='left',\
                            va='bottom') 
                    
                    axa.set_xlim(0,100)
                    axb.set_xlim(-20,10)
                    
                    axa.axvline(95,lw=3,ls='-.',c='navy')
                    dumstr = '95% RH'
                    axa.text(93,3.2,dumstr,fontsize=Fontsize,\
                            fontweight='bold',c='navy',ha='center',\
                            va='bottom') 
                    
                    
                    ax1.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(0.1,1.55),framealpha=0)
                    axa.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(0.6,1.7),framealpha=0)
                    axb.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(0.1,1.7),framealpha=0)                    
                    

                    
                    
                    #-------------------------------                    
                    # BASTA Reflectivity
                    #-------------------------------
                    dum_basta_ref = basta_ref_lim.copy()
                    dum_basta_ref[(dum_basta_ref > -999.) & (dum_basta_ref < -50.)] = -50
                    dum_basta_ref[dum_basta_ref > 20.] = 20.
                                 
                    
                    cmap = plt.cm.nipy_spectral
                    cmap.set_under('w')
                    cmap.set_bad('grey')
                    basta_bin_edges = np.arange(0,np.max(basta_height)+12.5+25.,25.)
                    ref_plot = ax2.pcolormesh(basta_time_dt_lim,\
                                                    basta_bin_edges/1.e3,\
                                                    dum_basta_ref[:,:-1],\
                                                    cmap=cmap,\
                                                    vmin=-50.1,vmax=20.1,\
                                                    shading='flat'
                                                   )                                         
                                 
                                 
                    # new ax with dimensions of the colorbar
                    cbar_ax1 = fig.add_axes([0.92,0.405,0.02,0.18])
                    dum_ticks = [-50,-40,-30,-20,-10,0,10,20]
                    cbar1 = fig.colorbar(ref_plot, cax=cbar_ax1,ticks=dum_ticks)       
                    cbar1.ax.set_ylabel('Reflectivity [dBZ]',fontsize=Fontsize)
                    cbar1.ax.tick_params(labelsize=Fontsize)
                    
                    
                    ax2.plot(basta_time_dt_lim,sonde_height_interp_final*1.e-3,\
                            lw=3,c='black',ls='solid',label='Sonde Height')
            
                    # CBH scatter
                    ax2.scatter(basta_time_dt_lim,merged_ceil_cbh_1_lim*1.e-3,s=10,\
                               color='magenta',label='CBH')            
            
            
                    #-------------------------------
                    # ARM Ceilometer
                    #-------------------------------
                                 
                    ax3.plot(basta_time_dt_lim,sonde_height_interp_final*1.e-3,\
                            lw=3,c='black',ls='solid',label='Sonde Height')
                    ax3.scatter(basta_time_dt_lim,merged_ceil_cbh_1_lim*1.e-3,s=10,\
                               color='magenta',label='CBH')        
                    
                    
                    ceil_backscatter_lim = np.squeeze(ceil_backscatter)
                    ceil_time_dt_lim = ceil_time_dt
                    tmpid = np.where(ceil_range <= y_height*1.e3)
                    tmpid = np.squeeze(tmpid)
                    ceil_backscatter_lim = ceil_backscatter_lim[:,:]
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
      

                    cmap = plt.cm.jet
                    #cmap = plt.cm.viridis
                    cmap.set_under('black')
                    cmap.set_bad('grey')
                    dum[dum == 0.] = np.nan
                    dum[dum < -8.] = -8.
                    dum[dum > -3.] = -3.

                    ceil_backscatter_plot = ax3.pcolormesh(ceil_time_dt_lim,\
                                                         ceil_range_lim*1.e-3,\
                                                         np.transpose(dum),\
                                                         cmap=cmap,\
                                                         vmin=-8,vmax=-3)                    
                    
                    
                    
                    cbar_ax3 = fig.add_axes([0.92,0.11,0.02,0.18])
                    # new ax with dimensions of the colorbar
                    dum_ticks = [-8,-7,-6,-5,-4,-3]
                    cbar3 = fig.colorbar(ceil_backscatter_plot, cax=cbar_ax3,ticks=dum_ticks)       
                    cbar3.ax.set_ylabel('$log_{10}$($\\beta_{att}$) [sr$^{-1}$ m$^{-1}$]',fontsize=Fontsize)
                    cbar3.ax.tick_params(labelsize=Fontsize)
                    
                    # CBH scatter
                    ax3.scatter(basta_time_dt_lim,merged_ceil_cbh_1_lim*1.e-3,s=10,\
                               color='magenta',label='CBH')                          
                    #-------------------------------
                    # Other stuff
                    #-------------------------------                            
                    ax1.text(-0.1,1.2,'(a)',fontsize=Fontsize*2.5,transform=ax1.transAxes)
                    ax2.text(-0.1,1.2,'(b)',fontsize=Fontsize*2.5,transform=ax2.transAxes)
                    ax3.text(-0.1,1.2,'(c)',fontsize=Fontsize*2.5,transform=ax3.transAxes)

     
                    ax1.xaxis.set_ticks_position("bottom")
                    ax1.yaxis.set_ticks_position("left")
                    ax2.xaxis.set_ticks_position("bottom")
                    ax2.yaxis.set_ticks_position("left")
                    ax3.xaxis.set_ticks_position("bottom")
                    ax3.yaxis.set_ticks_position("left")                    
                    
                    
                    ax1.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=1)
                    ax2.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=1)
                    ax3.grid(True,c='dimgrey',axis='both',which='both',ls='dotted',lw=1)
         
                    axa.grid(False)
                    axb.grid(False)
                                 
                    plt.subplots_adjust(wspace=0.375,hspace=0.7)
                    # Plot Date
                    tmp_time = current_date_sonde_times_dt[jj].strftime("%m/%d/%Y %H:%M")
                    plt.figtext(0.5,1.025,'Sounding Release Time:\n'+tmp_time+' UTC',\
                                fontsize=Fontsize*2,ha='center')
                    
                    fig_path = '/home/mwstanfo/figures/micre/sounding_rh_at_cbh/'
                    dum_time = current_date_sonde_times_dt[jj].strftime('%Y%m%d_%H%M')
                    outfile = 'sounding_rh_cbh_'+dum_time+'UTC_v2.png'
                    plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
                    #plt.show()
                    plt.close()    
                    #print(aaaa)
                                         
                                 
                

                #rh_at_cbh = current_sonde_rh_lim_interp[kk_ind]    
                if False:
                    
                    dfmt = mdates.DateFormatter('%H:%M')
                    fig = plt.figure(figsize=(6,8))
                    Fontsize=14
                    ax = fig.add_subplot(111)
                    axa = ax.twiny()
                    axb = ax.twiny()
                    ax.set_xlabel('Time',fontsize=Fontsize)
                    ax.set_ylabel('Height',fontsize=Fontsize)
                    ax.tick_params(labelsize=Fontsize)
                    ax.grid(which='both')
                    ax.plot(basta_time_dt_lim,sonde_height_interp_final*1.e-3,\
                            lw=3,c='black',ls='solid',label='Sonde Height')
                    ax.xaxis.set_major_formatter(dfmt)

                    axa.plot(current_sonde_rh_lim_interp,sonde_height_interp_final*1.e-3,\
                           lw=3,c='blue',ls='solid',label='RH') 
                    axb.plot(current_sonde_temperature_lim_interp-273.15,sonde_height_interp_final*1.e-3,\
                           lw=3,c='red',ls='solid',label='Temperature') 
                    #axa.set_ylim(0,4)
                    ax.set_ylim(0,9)
                    axa.set_ylim(0,9)
                    axb.set_ylim(0,9)
                    
                    # deal with axis colors
                    axa.spines['top'].set_color('blue')
                    axa.xaxis.label.set_color('blue')
                    axa.tick_params(axis='x', labelsize=Fontsize, colors='blue')
                    axa.spines['bottom'].set_visible(False)
                    axa.set_xlabel('RH [%]',fontsize=Fontsize)
                    axb.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
                    axb.xaxis.set_label_position("bottom")
                    axb.xaxis.set_ticks_position("bottom")
                    axb.spines["bottom"].set_position(('axes',-0.1))
                    axb.xaxis.label.set_color('red')
                    axb.tick_params(axis='x',labelsize=Fontsize,colors='red')
                    axb.spines['bottom'].set_color('red')
                    axb.spines['top'].set_color('blue')
                  
                    
                    # CBH scatter
                    ax.scatter(basta_time_dt_lim,merged_ceil_cbh_1_lim*1.e-3,s=10,\
                               color='magenta',label='CBH_1')
                    
                    axa.axhline(sonde_cbh*1.e-3,\
                              lw=3,c='green',ls='dashed',label='Sonde Loc. @ CBH_1')
                    axa.axvline(rh_at_cbh,\
                              lw=3,c='deepskyblue',ls='dashed',label='RH @ CBH_1') 
                    dumstr = str(np.around(rh_at_cbh,1))+'%'
                    axa.text(107,0.,dumstr,fontsize=Fontsize*1.5,\
                            fontweight='bold',c='deepskyblue',ha='left',\
                            va='bottom') 
                    axa.set_xlim(0,105)
                    
                    ax.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(-0.05,1.2))
                    axa.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(0.6,1.25))
                    axb.legend(loc='upper left',fontsize=Fontsize,bbox_to_anchor=(-0.05,1.275))

                    #ax.set_xlim(basta_time_dt_lim[15],basta_time_dt_lim[25])
                    #plt.show()                        
                        
                    # Plot Date
                    tmp_time = current_date_sonde_times_dt[jj].strftime("%m/%d/%Y %H:%M")
                    plt.figtext(0.5,1.1,'Sounding Release Time:\n'+tmp_time+' UTC',\
                                fontsize=Fontsize*2,ha='center')
                    #print(aaaaa)
                    axa.grid(False)
                    axb.grid(False)
                    plt.show()
                    #plt.close()

                    
                    fig_path = '/home/mwstanfo/figures/sounding_rh_cbh/'
                    dum_time = current_date_sonde_times_dt[jj].strftime('%Y%m%d_%H%M')
                    outfile = 'sounding_rh_cbh_'+dum_time+'UTC_v2.png'
                    #plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
                    plt.close()    
                    
                    
                    
                    
                        
                # test interpolation
                if False:
                    dfmt = mdates.DateFormatter('%H:%M')
                    fig = plt.figure(figsize=(6,8))
                    Fontsize=14
                    ax = fig.add_subplot(111)
                    ax.set_xlabel('Time',fontsize=Fontsize)
                    ax.set_ylabel('Height [km]',fontsize=Fontsize)
                    ax.tick_params(labelsize=Fontsize)
                    ax.grid(which='both')
                    ax.plot(current_sonde_time_dt_long_lim,current_sonde_height_lim,\
                            lw=3,c='black',ls='solid',label='Orig')
                    ax.plot(basta_time_dt_lim,current_sonde_height_lim_interp*1.e-3,\
                           lw=5,c='deepskyblue',ls='dashed',label='Interp_1')
                    ax.plot(basta_time_dt_lim,sonde_height_interp_final*1.e-3,\
                           lw=7,c='red',ls='dotted',label='Interp 2')                                                                
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.legend(loc='upper left',fontsize=Fontsize)
                    plt.show()
                    
    #print(aaaa)
    #===========================================
    # End RH @ CBH
    #===========================================
    ii+=1
    
print(aaaa)                                                                         
                                                                           
sonde_rh_cbh_all = np.array(sonde_rh_cbh_all)
sonde_time_all = np.array(sonde_time_all)
sonde_temp_all = np.array(sonde_temp_all)
sonde_rh_all = np.array(sonde_rh_all)  

dumid = np.where(sonde_rh_all > 98.)
print(np.size(dumid)/np.size(sonde_rh_all)*100.)
dumid = np.where(sonde_rh_all > 95.)
print(np.size(dumid)/np.size(sonde_rh_all)*100.)
dumid = np.where(sonde_rh_all > 90.)
print(np.size(dumid)/np.size(sonde_rh_all)*100.)

# From First Try
#41.70403587443946
#66.3677130044843
#79.82062780269058

#===========================================
# Plot 2D Histogram of CBH RH & Temperature
#===========================================
sns.set_theme()
#sns.set_style('dark')
sns.set_style('ticks')
sns.set(rc={'axes.facecolor':'white','axes.edgecolor': 'black','grid.color':'grey'})
sns.set_context('talk')  

fig = plt.figure(figsize=(8.3,8.3))
ax1 = fig.add_subplot(111)
Fontsize=18
t_bins = np.arange(-36,12,2)
rh_bins = np.arange(50,101,1)
hist_plot = ax1.hist2d(sonde_temp_all,sonde_rh_all,\
                       bins=[t_bins,rh_bins],cmap='inferno',cmin=1)

ax1.grid(which='both',ls='dotted',c='grey')
ax1.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
ax1.set_ylabel('RH [%]',fontsize=Fontsize)
ax1.tick_params(labelsize=Fontsize)

# new ax with dimensions of the colorbar
cbar_ax1 = fig.add_axes([0.92,0.125,0.03,0.75])
cbar1 = fig.colorbar(hist_plot[3], cax=cbar_ax1)       
cbar1.ax.set_ylabel('Count',fontsize=Fontsize)
cbar1.ax.tick_params(labelsize=Fontsize)

#ax1.axhline(95,c='darkorange',lw=3,ls='dashed')
#ax1.text(-0.1,0.9,'95%',transform=ax1.transAxes,fontsize=Fontsize,fontweight='bold',color='darkorange',ha='left',va='center')

ax1.xaxis.set_ticks_position("bottom")
ax1.yaxis.set_ticks_position("left")
ax1.grid(True,which='both',axis='both',c='grey',ls='solid')
#plt.show()

fig_path = '/home/mwstanfo/figures/micre_paper/'
outfile = 'fig_c1.png'
#outfile = 'fig_b1.eps'
plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
#plt.show()
plt.close()

print(aaaa)                                                                           
                                                                           
                                                                           
if False:                                                                           

    #===========================================
    # Start Plot
    #===========================================        
    for jj in range(num_current_soundings):

        time_delta = datetime.timedelta(hours=1)
        start_time = sonde_time_dt[jj]
        #end_time = target_end_time
        end_time = sonde_time_dt[jj]+datetime.timedelta(minutes=30)
        
        dumid = np.where((basta_time_dt >= start_time) & (basta_time_dt <= end_time))
        basta_time_lim = basta_time_dt[dumid]

        if np.size(basta_time_lim) == 0.:
            basta_present_flag = False
        else:
            basta_present_flag = True

        
        
        # Find cloud layers in RH profile
        #current_sonde_rh = sonde_rh_interp[jj]
        current_sonde_rh = sonde_rh_interp[jj]
        # mask RH
        cloud_layer_mask = np.zeros(len(current_sonde_rh))

        dumid = np.where(current_sonde_rh >= 98.)
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
        Fontsize=14
        fig = plt.figure(figsize=(13,25))  
        gs=GridSpec(5,3) # 6 rows, 4 columns
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])       
        ax4 = fig.add_subplot(gs[1,:])  
        ax5 = fig.add_subplot(gs[2,:]) 
        ax6 = fig.add_subplot(gs[3,:])      
        ax7 = fig.add_subplot(gs[4,:])
        
        axlist = [ax1,ax2,ax4,ax5,ax6,ax7]
        for ax in axlist:
            ax.tick_params(labelsize=Fontsize)
            ax.grid(which='both',c='grey')
            
        axlist2 = [ax1,ax2]
        for ax in axlist2:
            ax.set_ylabel('Height [km]',fontsize=Fontsize)
            ax.set_ylim(0,y_height)
                   
        ax1.set_xlabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize)
        ax2.set_xlabel('RH [%]',fontsize=Fontsize)
 
        #------------------------------
        # plot temperature, q, & theta
        #------------------------------
        dumid = np.where(sonde_height[jj] <= y_height)
        tmp_height = sonde_height[jj][dumid]
        tmp_temp = sonde_temperature[jj][dumid]-273.15
        ax1.plot(tmp_temp,tmp_height,lw=2,c='red',ls='solid')
        
        # deal with axis colors
        ax1.spines['bottom'].set_color('red')
        ax1.xaxis.label.set_color('red')
        ax1.tick_params(axis='x', colors='red')


        # add zero degree isotherm
        zero_temp,zero_height_id = find_nearest(tmp_temp,0)
        ax1.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon',label='0$^{\\circ}$C')
        ax1.legend(fontsize=Fontsize,loc='upper center',bbox_to_anchor=(0.5,1.25))
        
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
        ax2.axvline(98.,c='black',ls='dashed',lw=2)
        
        lgnd99=ax2.legend(fontsize=Fontsize,loc='upper left',bbox_to_anchor=(1.,1.215))
        
        
        if num_cloud_objects > 0.:
            dumx=ax2.get_xlim()
            dumx = np.arange(dumx[0],dumx[1]+1)
            for kk in range(num_cloud_objects):
                ax2.fill_between(dumx,sonde_cbhs[kk]*1.e-3,sonde_cths[kk]*1.e-3,facecolor='purple',alpha=0.25)
                ax2.axhline(sonde_cbhs[kk]*1.e-3,lw=2,c='purple')
                ax2.axhline(sonde_cths[kk]*1.e-3,lw=2,c='purple')
            
        black_patch = mpatches.Patch(color='purple',alpha=0.25,label='Cloud Layer')

        lgnd98 = ax2.legend(handles=[black_patch],\
                            fontsize=Fontsize,\
                            bbox_to_anchor=(1.,0.85),\
                            ncol=1,loc='upper left') 
        

        
        legend_elements = [Line2D([0], [0],color='black',ls='dashed',lw=2,label='98% RH')]
        lgnd97=ax2.legend(handles=legend_elements, loc='upper left',\
               ncol=1,fontsize=Fontsize,bbox_to_anchor=(1.,0.65)) 
        
        
        if basta_present_flag == True:
            # Now find the average CBH 1 within 5 minutes of
            # release time
            dum_time_delta = datetime.timedelta(minutes=5)
            dum_time_id = np.where((basta_time_lim >= sonde_time_dt[jj]) & (basta_time_lim <= sonde_time_dt[jj]+dum_time_delta) )
            dum_merged_ceil_cbh_1_lim = merged_ceil_cbh_1_lim[dum_time_id]
            mean_ceil_cbh_1 = np.nanmean(dum_merged_ceil_cbh_1_lim)
            ceil_cbh_1_10th = np.nanpercentile(dum_merged_ceil_cbh_1_lim,10)
            ceil_cbh_1_90th = np.nanpercentile(dum_merged_ceil_cbh_1_lim,90)
            
            ax2.axhline(mean_ceil_cbh_1/1.e3,c='darkgreen',lw=4,ls='dotted')
            ax2.axhline(ceil_cbh_1_10th/1.e3,c='deeppink',lw=2,ls='dashed')
            ax2.axhline(ceil_cbh_1_90th/1.e3,c='deeppink',lw=2,ls='dashed')            
            
        elif ceilometer_present == True:
            # Now find the average CBH 1 within 5 minutes of
            # release time
            dum_time_delta = datetime.timedelta(minutes=5)
            dum_time_id = np.where((ceil_time_dt >= sonde_time_dt[jj]) & (ceil_time_dt <= sonde_time_dt[jj]+dum_time_delta) )
            if np.size(dum_time_id) > 0.:
                dum_ceil_cbh_1_lim = ceil_cbh_1[dum_time_id]
                mean_ceil_cbh_1 = np.nanmean(dum_ceil_cbh_1_lim)
                ceil_cbh_1_10th = np.nanpercentile(dum_ceil_cbh_1_lim,10)
                ceil_cbh_1_90th = np.nanpercentile(dum_ceil_cbh_1_lim,90)        

                ax2.axhline(mean_ceil_cbh_1/1.e3,c='darkgreen',lw=4,ls='dotted')
                ax2.axhline(ceil_cbh_1_10th/1.e3,c='deeppink',lw=2,ls='dashed')
                ax2.axhline(ceil_cbh_1_90th/1.e3,c='deeppink',lw=2,ls='dashed')
                
            elif aad_ceilometer_present == True:
                # Now find the average CBH 1 within 5 minutes of
                # release time
                dum_time_delta = datetime.timedelta(minutes=5)
                dum_time_id = np.where((aad_ceil_time_dt >= sonde_time_dt[jj]) & (aad_ceil_time_dt <= sonde_time_dt[jj]+dum_time_delta) )
                if np.size(dum_time_id) > 0.:
                    dum_ceil_cbh_1_lim = aad_ceil_cbh_1[dum_time_id]
                    mean_ceil_cbh_1 = np.nanmean(dum_ceil_cbh_1_lim)
                    ceil_cbh_1_10th = np.nanpercentile(dum_ceil_cbh_1_lim,10)
                    ceil_cbh_1_90th = np.nanpercentile(dum_ceil_cbh_1_lim,90)        

                    ax2.axhline(mean_ceil_cbh_1/1.e3,c='darkgreen',lw=4,ls='dotted')
                    ax2.axhline(ceil_cbh_1_10th/1.e3,c='deeppink',lw=2,ls='dashed')
                    ax2.axhline(ceil_cbh_1_90th/1.e3,c='deeppink',lw=2,ls='dashed')                
                
        
        dumstr = 'CEIL CBH 5-min. Avg.\nafter release'
        dumstr2 = 'CEIL CBH 5-min. 10$^{th}$/ 90$^{th}$\npercentile after release'
        legend_elements = [Line2D([0], [0],color='darkgreen',ls='dotted',lw=4,label=dumstr),\
                           Line2D([0], [0],color='deeppink',ls='dashed',lw=2,label=dumstr2)]
        lgnd96=ax2.legend(handles=legend_elements, loc='upper left',\
               ncol=1,fontsize=Fontsize*0.8,bbox_to_anchor=(1.,0.44))       
        
        ax2.add_artist(lgnd99)
        ax2.add_artist(lgnd98)
        ax2.add_artist(lgnd97)
            

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
                        
            
            
            basta_ref_lim[(basta_ref_lim > -999.) & (basta_ref_lim < -50.)] = -50.
            basta_ref_lim[basta_ref_lim > 20.] = 20.
            cmap = plt.cm.nipy_spectral
            cmap.set_under('w')
            cmap.set_bad('grey')
            basta_bin_edges = np.arange(0,np.max(basta_height_lim)+12.5+25.,25.)
            ref_plot = ax4.pcolormesh(basta_time_lim,\
                                            basta_bin_edges/1.e3,\
                                            basta_ref_lim[:,:-1],\
                                            cmap=cmap,\
                                            vmin=-50.1,vmax=20.1,\
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
                    s=10,marker='o',color='black',label='CBH 1')
                cbh_1_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_1_lim*1.e-3,\
                    s=10,marker='o',color='black',label='CBH 1')
            if ~np.isnan(np.nanmax(merged_ceil_cbh_2_lim)):
                cbh_2_plot = ax4.scatter(basta_time_lim,merged_ceil_cbh_2_lim*1.e-3,\
                    s=10,marker='o',color='magenta',label='CBH 2')
                cbh_2_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_2_lim*1.e-3,\
                    s=10,marker='o',color='magenta',label='CBH 2')
            if ~np.isnan(np.nanmax(merged_ceil_cbh_3_lim)):
                cbh_3_plot = ax4.scatter(basta_time_lim,merged_ceil_cbh_3_lim*1.e-3,\
                    s=10,marker='o',color='aqua',label='CBH 3')                  
                cbh_3_plot = ax5.scatter(basta_time_lim,merged_ceil_cbh_3_lim*1.e-3,\
                    s=10,marker='o',color='aqua',label='CBH 3')  
                
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
            ax4.grid(which='both',c='grey')
            ax5.grid(which='both',c='grey')

            # new ax with dimensions of the colorbar
            cbar_ax1 = fig.add_axes([0.92,0.615,0.02,0.1])
            dum_ticks = [-50,-40,-30,-20,-10,0,10,20]
            cbar1 = fig.colorbar(ref_plot, cax=cbar_ax1,ticks=dum_ticks)       
            cbar1.ax.set_ylabel('Reflectivity [dBZ]',fontsize=Fontsize)
            cbar1.ax.tick_params(labelsize=Fontsize)  
            
            # new ax with dimensions of the colorbar
            cbar_ax2 = fig.add_axes([0.92,0.4525,0.02,0.1])
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
                

            # add zero degree isotherm
            zero_temp,zero_height_id = find_nearest(tmp_temp,0)
            ax4.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon')
            ax5.axhline(tmp_height[zero_height_id],lw=2,ls='dashed',color='maroon')
            
            legend_elements = [Line2D([0],[0],lw=2,color='maroon',ls='dashed',label='0$^{\\circ}$C')]            
            lgnd123=ax4.legend(handles=legend_elements, loc='upper center',\
                   ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.85,1.25))  
            
                
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
                                bbox_to_anchor=(0.375,1.25),\
                                ncol=2,loc='upper center') 
            lgnd998 = ax4.legend(handles=legend_elements2,\
                                fontsize=Fontsize,\
                                bbox_to_anchor=(0.625,1.25),\
                                ncol=2,loc='upper center') 
            
            grey_patch = mpatches.Patch(color='grey',label='Bad Radar Data')
            
            lgnd = ax5.legend(handles=[grey_patch],\
                                fontsize=Fontsize,\
                                bbox_to_anchor=(0.85,1.35),\
                                ncol=1,loc='upper center')            

            legend_elements = [Line2D([0], [0], marker='.', color='w', label='CBH 1',
                          markerfacecolor='black', markersize=15),\
                              Line2D([0], [0], marker='.', color='w', label='CBH 2',
                          markerfacecolor='magenta', markersize=15),\
                              Line2D([0], [0], marker='.', color='w', label='CBH 3',
                          markerfacecolor='aqua', markersize=15)]
            lgnd3=ax4.legend(handles=legend_elements, loc='lower left',\
                   ncol=3,fontsize=Fontsize,bbox_to_anchor=(0.,-0.5))             

            legend_elements = [Line2D([0], [0],color='black',ls='solid',lw=3,label='Sonde Path')]
            lgnd4=ax4.legend(handles=legend_elements, loc='lower center',\
                   ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.575,-0.5))  
        
            
            
            
            
            ax4.add_artist(lgnd3)
            ax4.add_artist(lgnd999)
            ax4.add_artist(lgnd998)
            ax4.add_artist(lgnd123)

            
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
                
                
                
                cmap = plt.cm.jet
                cmap.set_under('navy')
                cmap.set_bad('grey')
                dum[dum < -8.] = -8.
                dum[dum > -3.] = -3.

                ceil_backscatter_plot = ax6.pcolormesh(ceil_time_dt_lim,\
                                                     ceil_range_lim*1.e-3,\
                                                     np.transpose(dum),\
                                                     cmap=cmap,
                                                     vmin=-8,vmax=-3)

                if ~np.isnan(np.nanmax(ceil_cbh_1_native_lim)):
                    cbh_1_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_1_native_lim*1.e-3,\
                        s=10,marker='o',color='black',label='CBH 1')
                if ~np.isnan(np.nanmax(ceil_cbh_2_native_lim)):
                    cbh_2_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_2_native_lim*1.e-3,\
                        s=10,marker='o',color='magenta',label='CBH 2')
                if ~np.isnan(np.nanmax(ceil_cbh_3_native_lim)):
                    cbh_3_plot = ax6.scatter(ceil_time_dt_lim,ceil_cbh_3_native_lim*1.e-3,\
                        s=10,marker='o',color='aqua',label='CBH 3')                    
                    
                cbar_ax3 = fig.add_axes([0.92,0.29,0.02,0.1])
                # new ax with dimensions of the colorbar
                dum_ticks = [-8,-7,-6,-5,-4,-3]
                cbar3 = fig.colorbar(ceil_backscatter_plot, cax=cbar_ax3,ticks=dum_ticks)       
                cbar3.ax.set_ylabel('$log_{10}$($\\beta_{att}$) [sr$^{-1}$ m$^{-1}$]',fontsize=Fontsize)
                cbar3.ax.tick_params(labelsize=Fontsize)
                ax6.grid(which='both',c='grey')
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
                
                
                
                
                legend_elements = [Line2D([0], [0], marker='.', color='w', label='CBH 1',
                              markerfacecolor='black', markersize=15),\
                                  Line2D([0], [0], marker='.', color='w', label='CBH 2',
                              markerfacecolor='magenta', markersize=15),\
                                  Line2D([0], [0], marker='.', color='w', label='CBH 3',
                              markerfacecolor='aqua', markersize=15)]
                lgnd2=ax6.legend(handles=legend_elements, loc='lower right',\
                       ncol=3,fontsize=Fontsize,bbox_to_anchor=(1.1,-0.4))

                grey_patch = mpatches.Patch(color='grey',label='Missing CEIL Data')
                lgnd888 = ax6.legend(handles=[grey_patch],\
                                    fontsize=Fontsize,\
                                    bbox_to_anchor=(0.15,1.25),\
                                    ncol=1,loc='upper center')                     
  
                legend_elements = [Line2D([0], [0],color='black',ls='solid',lw=3,label='Sonde Path')]
                lgnd5=ax6.legend(handles=legend_elements, loc='lower center',\
                       ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.1,-0.4))  



                ax6.add_artist(lgnd2)                
                ax6.add_artist(lgnd888)                
                
                
        else:
            ax6.set_xlim(0,1)
            ax6.set_ylim(0,1)
            tmpplot=ax6.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
            ax6.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
            tmpplot[0].axes.get_xaxis().set_visible(False)
            tmpplot[0].axes.get_yaxis().set_visible(False)    
            ax6.set_title('No ARM Ceilometer Data Available',fontsize=Fontsize*2.,c='red') 

        #-----------------------------------
        # Plot Univ. of Canterbury Ceilometer
        #-----------------------------------
        if aad_ceilometer_present == True:

            
            dumid = np.where((aad_ceil_time_dt >= start_time) & (aad_ceil_time_dt <= end_time))
            if np.size(dumid) == 0.:
                ax6.set_xlim(0,1)
                ax6.set_ylim(0,1)
                tmpplot=ax6.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
                ax6.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
                tmpplot[0].axes.get_xaxis().set_visible(False)
                tmpplot[0].axes.get_yaxis().set_visible(False)    
                ax6.set_title('No Univ. of Canterbury Ceilometer Data Available',fontsize=Fontsize*2.,c='red')             
            else:
                ax7.set_ylabel('Height [km]',fontsize=Fontsize)
                ax7.set_ylim(0,y_height)
                ax7.set_xlabel('UTC Time [HH:MM]',fontsize=Fontsize)
                ax7.plot(sonde_time_dt_long[jj],sonde_height[jj],lw=3,c='k',ls='solid')
                ax7.xaxis.set_major_formatter(dfmt)
                ax7.set_xlim(start_time,end_time) 
                dumid = np.squeeze(dumid)
                aad_ceil_cbh_1_native_lim = aad_ceil_cbh_1[dumid]
                aad_ceil_cbh_2_native_lim = aad_ceil_cbh_2[dumid]
                aad_ceil_cbh_3_native_lim = aad_ceil_cbh_3[dumid]
                aad_ceil_backscatter_lim = aad_ceil_backscatter[dumid,:]
                aad_ceil_backscatter_lim = np.squeeze(aad_ceil_backscatter_lim)
                aad_ceil_time_dt_lim = aad_ceil_time_dt[dumid]
                tmpid = np.where(aad_ceil_height <= y_height*1.e3)
                tmpid = np.squeeze(tmpid)
                aad_ceil_backscatter_lim = aad_ceil_backscatter_lim[:,tmpid]
                aad_ceil_range_lim = aad_ceil_height[tmpid]
                aad_ceil_backscatter_lim = aad_ceil_backscatter_lim*1.e-3 
                
                dumid = np.where((~np.isnan(aad_ceil_backscatter_lim)) & (aad_ceil_backscatter_lim > 0.) )
                dum = aad_ceil_backscatter_lim.copy()
                dum[dumid] = np.log10(dum[dumid])
                dumid = np.where(aad_ceil_backscatter_lim == 0.)
                dum[dumid] = np.nan
                dumid = np.where(aad_ceil_backscatter_lim < 0.)
                dum[dumid] = np.nan
                dum[np.isnan(dum)] = 0.
                
                
                dum_time_delta = datetime.timedelta(seconds=60)
                start_time_ts = toTimestamp(start_time)
                end_time_ts = toTimestamp(end_time)
                dumx = np.arange(start_time_ts,end_time_ts,1)
                dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])
                
                
                dum_diff = np.diff(aad_ceil_time_dt_lim)
                dum_diff_0 = np.array([datetime.timedelta(seconds=0)])
                dum_diff = np.concatenate((dum_diff,dum_diff_0))
                #uniq_diff = np.unique(dum_diff)
                dumid = np.where(dum_diff > datetime.timedelta(seconds=30))
                if np.size(dumid) > 0.:
                    dum[dumid] = np.nan          
                

                cmap = plt.cm.jet
                cmap.set_under('navy')
                cmap.set_bad('grey')                

                dum[dum < -8.] = -8.
                dum[dum > -3.] = -3.
                aad_ceil_backscatter_plot = ax7.pcolormesh(aad_ceil_time_dt_lim,\
                                                     aad_ceil_range_lim*1.e-3,\
                                                     np.transpose(dum),\
                                                     cmap=cmap,
                                                     vmin=-8,vmax=-3)  

                if ~np.isnan(np.nanmax(aad_ceil_cbh_1_native_lim)):
                    cbh_1_plot = ax7.scatter(aad_ceil_time_dt_lim,aad_ceil_cbh_1_native_lim*1.e-3,\
                        s=10,marker='o',color='black',label='CBH 1')
                if ~np.isnan(np.nanmax(aad_ceil_cbh_2_native_lim)):
                    cbh_2_plot = ax7.scatter(aad_ceil_time_dt_lim,aad_ceil_cbh_2_native_lim*1.e-3,\
                        s=10,marker='o',color='magenta',label='CBH 2')
                if ~np.isnan(np.nanmax(aad_ceil_cbh_3_native_lim)):
                    cbh_3_plot = ax7.scatter(aad_ceil_time_dt_lim,aad_ceil_cbh_3_native_lim*1.e-3,\
                        s=10,marker='o',color='aqua',label='CBH 3')    
            
            
                dum_time_delta = datetime.timedelta(seconds=60)
                start_time_ts = toTimestamp(start_time)
                end_time_ts = toTimestamp(end_time)
                dumx = np.arange(start_time_ts,end_time_ts,1)
                dumx_dt = np.array([toDatetime(dumx[dd]) for dd in range(len(dumx))])

                dumid = np.where(dumx_dt > aad_ceil_time_dt_lim[-1])
                if np.size(dumid) > 0.:
                    ax7.fill_between(dumx_dt[dumid],0,y_height,facecolor='grey')

                dumid = np.where(dumx_dt < aad_ceil_time_dt_lim[0])
                if np.size(dumid) > 0.:
                    ax7.fill_between(dumx_dt[dumid],0,y_height,facecolor='grey')
                
                            
            
                cbar_ax4 = fig.add_axes([0.92,0.125,0.02,0.1])
                # new ax with dimensions of the colorbar
                dum_ticks = [-8,-7,-6,-5,-4,-3]
                cbar4 = fig.colorbar(aad_ceil_backscatter_plot, cax=cbar_ax4,ticks=dum_ticks)       
                cbar4.ax.set_ylabel('$log_{10}$($\\beta_{att}$) [sr$^{-1}$ m$^{-1}$]',\
                                    fontsize=Fontsize)
                cbar4.ax.tick_params(labelsize=Fontsize)
                ax7.grid(which='both',c='grey')      
                ax7.set_title('Univ. of Canterbury Ceilometer',fontsize=Fontsize*1.5)
            
            
                legend_elements = [Line2D([0], [0], marker='.', color='w', label='CBH 1',
                              markerfacecolor='black', markersize=15),\
                                  Line2D([0], [0], marker='.', color='w', label='CBH 2',
                              markerfacecolor='magenta', markersize=15),\
                                  Line2D([0], [0], marker='.', color='w', label='CBH 3',
                              markerfacecolor='aqua', markersize=15)]
                lgnd222=ax7.legend(handles=legend_elements, loc='lower right',\
                       ncol=3,fontsize=Fontsize,bbox_to_anchor=(1.1,-0.4))

                grey_patch = mpatches.Patch(color='grey',label='Missing CEIL Data')
                lgnd333 = ax7.legend(handles=[grey_patch],\
                                    fontsize=Fontsize,\
                                    bbox_to_anchor=(0.15,1.25),\
                                    ncol=1,loc='upper center')                     
  
                legend_elements = [Line2D([0], [0],color='black',ls='solid',lw=3,label='Sonde Path')]
                lgnd555=ax7.legend(handles=legend_elements, loc='lower center',\
                       ncol=1,fontsize=Fontsize,bbox_to_anchor=(0.1,-0.4))  



                ax7.add_artist(lgnd222)                
                ax7.add_artist(lgnd333)   
        else:
            ax7.set_xlim(0,1)
            ax7.set_ylim(0,1)
            tmpplot=ax7.plot(np.arange(0,1.25,0.25),np.arange(0,1.25,0.25),lw=2,c='k')
            ax7.plot(np.arange(0,1.25,0.25),np.flip(np.arange(0,1.25,0.25)),lw=2,c='k')
            tmpplot[0].axes.get_xaxis().set_visible(False)
            tmpplot[0].axes.get_yaxis().set_visible(False)    
            ax7.set_title('No Univ. of Canterbury Ceilometer Data Available',\
                          fontsize=Fontsize*2.,c='red')            

            
        plt.subplots_adjust(wspace=0.35,hspace=0.6)
        
        # Plot Date
        tmp_time = current_date_sonde_times_dt[jj].strftime("%m/%d/%Y %H:%M")
        plt.figtext(0.5,0.9,'Sounding Release Time:\n'+tmp_time+' UTC',\
                    fontsize=Fontsize*2,ha='center')
        
        if num_cloud_objects > 0.:
            dumtext = '# RH Cloud Layers: {}'.format(str(num_cloud_objects))
            plt.figtext(0.7,0.925,dumtext,fontsize=Fontsize,ha='left',c='peru')     

            dumtext = 'RH CTHs: {} m'.format(sonde_cths)
            plt.figtext(0.7,0.9125,dumtext,fontsize=Fontsize,ha='left',c='peru')     

            dumtext = 'RH CBHs: {} m'.format(sonde_cbhs)
            plt.figtext(0.7,0.9,dumtext,fontsize=Fontsize,ha='left',c='peru')     
        else:
            dumtext = 'No RH Cloud Layers'
            plt.figtext(0.7,0.925,dumtext,fontsize=Fontsize,ha='left',c='peru') 
        

        
        #fig_path = '/home/mwstanfo/figures/sounding_figs/'
        #dum_time = current_date_sonde_times_dt[jj].strftime('%Y%m%d_%H%M')
        #outfile = 'sounding_summary_'+dum_time+'UTC.png'
        #plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
        #plt.close()
        plt.show()
        plt.close()
        
        print(aaaa)
    print(aaaa)
    #if ii == 30:
    #    print(aaaaa)
    #===========================================
    # End Plot
    #===========================================
    
    ii+=1
    continue
    print(aaaa)    

    
    
    
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    # Mask BASTA times according to sounding sufficiency
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NEW!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #sonde_insufficient_mask = np.zeros(len(basta_time_dt))
    sonde_sufficient_mask = np.zeros(len(basta_time_dt))
    dum_range = np.arange(0,num_soundings,1)
    target_diff = datetime.timedelta(hours=6,minutes=30)
    # in this approach, if "sonde_sufficient_mask"
    # is nonzero, then one of the possible soundings
    # is within 6 hours of it, and we'll keeep those values.
    # If interpolation is possible, then that is used, otherwise
    # constant values are used.
    for iii in range(num_soundings):
        diff = basta_time_dt - sonde_time_dt[iii]
        diff = np.array([diff[dd].total_seconds() for dd in range(len(diff))])
        diff = np.abs(diff)
        dumid = np.where(diff <= target_diff.total_seconds())
        if np.size(dumid) > 0.:
            dumid = np.squeeze(dumid)
            #sonde_insufficient_mask[dumid] = iii+1
            sonde_sufficient_mask[dumid] = iii+1
            
    sonde_insufficient_mask = np.zeros(len(basta_time_dt))
    dumid = np.where(sonde_sufficient_mask == 0.)
    sonde_insufficient_mask[dumid] = 1
    #plt.plot(basta_time_dt,sonde_insufficient_mask)
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Legacy artifact; keeping for documentation
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False:
    #if True:
        # if either the before or after sonde is insufficient, then mask the
        if (before_sonde_insufficient_flag == True) or (after_sonde_insufficient_flag == True):
            # if only the before sounding is insufficient, mask only times before the first availble
            # sounding
            if (before_sonde_insufficient_flag == True) and (after_sonde_insufficient_flag == False):
            # if only the before sounding is insufficeint, mask only times before the first available
            # sounding
                tmpid = np.where(basta_time_dt < sonde_time_dt[0])
                tmpid = np.squeeze(tmpid)
                sonde_insufficient_mask[tmpid] = 1

            elif (before_sonde_insufficient_flag == False) and (after_sonde_insufficient_flag == True):
                # if only the after sounding is insufficient, mask only times after the last available
                # sounding
                tmpid = np.where(basta_time_dt > sonde_time_dt[-1])
                tmpid = np.squeeze(tmpid)
                sonde_insufficient_mask[tmpid] = 1

            # if both the before and after soundings are insufficient for interpolation
            # This means that one or two soundings are available on the date. If two are
            # available, we'll interpolate in between these times and save this data
            # If only one is available: Here we probably don't want to completely disregard the sounding.
            # if there is only one sounding available on the day, such as at 12Z, we probably
            # want to keep some window around the sounding, say 3 hours on either side. 
            # if the only available sounding on a given day is the around 23Z, then we'd want
            # to keep 3 hours prior to that sounding as well as 3 hours past the previous sounding
            # if the previous sounding was at 23Z on the day prior.
            elif (before_sonde_insufficient_flag == True) and (after_sonde_insufficient_flag == True):
                # black out periods
                if num_soundings == 2:
                    tmpid = np.where(basta_time_dt < sonde_time_dt[0])
                    if np.size(tmpid) > 0.:
                        tmpid = np.squeeze(tmpid)
                        sonde_insufficient_mask[tmpid] = 1
                    tmpid = np.where(basta_time_dt > sonde_time_dt[-1])
                    if np.size(tmpid) > 0.:
                        tmpid = np.squeeze(tmpid)
                        sonde_insufficient_mask[tmpid] = 1
                elif num_soundings == 1:
                    # If only one sounding on the day and there isn't a prior or before sounding,
                    # then preserve 3 hours around that one sounding
                    time_delta_3hr_prior = sonde_time_dt[0] - datetime.timedelta(hours=3)
                    time_delta_3hr_after = sonde_time_dt[0] + datetime.timedelta(hours=3)

                    tmpid = np.where((basta_time_dt < time_delta_3hr_prior) | (basta_time_dt > time_delta_3hr_after))
                    sonde_insufficient_mask[tmpid] = 1

                    # in the "interp_2" sounding variables, fill 3 hrs on each side of the sounding time
                    # to the sounding values
                    sonde_temperature_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_rh_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_rh_i_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_q_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_u_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_v_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_wind_speed_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_wind_dir_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_theta_interp_2 = np.zeros(np.shape(basta_ref))-999.
                    sonde_theta_e_interp_2 = np.zeros(np.shape(basta_ref))-999.

                    tmpid = np.where(sonde_insufficient_mask == 0)
                    tmpid = np.squeeze(tmpid)
                    for kk in range(len(tmpid)):
                        sonde_temperature_interp_2[:,tmpid[kk]] = np.squeeze(sonde_temperature_interp)
                        sonde_rh_interp_2[:,tmpid[kk]] = np.squeeze(sonde_rh_interp)
                        sonde_rh_i_interp_2[:,tmpid[kk]] = np.squeeze(sonde_rh_i_interp)
                        sonde_q_interp_2[:,tmpid[kk]] = np.squeeze(sonde_q_interp)
                        sonde_u_interp_2[:,tmpid[kk]] = np.squeeze(sonde_u_interp)
                        sonde_v_interp_2[:,tmpid[kk]] = np.squeeze(sonde_v_interp)
                        sonde_theta_interp_2[:,tmpid[kk]] = np.squeeze(sonde_theta_interp)
                        sonde_theta_e_interp_2[:,tmpid[kk]] = np.squeeze(sonde_theta_e_interp)
                        sonde_wind_speed_interp_2[:,tmpid[kk]] = np.squeeze(sonde_wind_speed_interp)
                        sonde_wind_dir_interp_2[:,tmpid[kk]] = np.squeeze(sonde_wind_dir_interp)
                else:
                    raise RuntimeError("Something's wrong.")
            

    
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    # End sounding insufficient data masking
    #------------------------------------------------------------------    
    #------------------------------------------------------------------    
    #------------------------------------------------------------------    


    #===========================================
    # Begin interpolated cluster ID block
    #===========================================
    cluster_id_interp = np.zeros(len(basta_time_dt))-999.
    
    # Loop through cluster (sounding) times
    for iii in range(len(current_cluster_times_dt)):
        #print(current_cluster_times_dt[iii],current_cluster_ids[iii])
        tmp_diff_time = datetime.timedelta(hours=6)
        if current_cluster_times_dt[iii] != -999.:
            tmp_before_time = current_cluster_times_dt[iii]-tmp_diff_time
            tmp_after_time = current_cluster_times_dt[iii]+tmp_diff_time
            tmpid = np.where( (basta_time_dt >= tmp_before_time) & (basta_time_dt < tmp_after_time) )
            if np.size(tmpid) > 0.:
                cluster_id_interp[tmpid] = current_cluster_ids[iii]


    #===========================================
    # End interpolated cluster ID block
    #===========================================

    

    

    
    #===========================================
    # Begin Satellite Block
    #===========================================
    tmpid = np.where(sat_dates_dt == date)
    if np.size(tmpid) == 0.:
        sat_present = False
        #raise RuntimeError('No satellite files at all for this date.')
    elif np.size(tmpid) > 0.:
        print('Begin satellite processing.')                    
        
        sat_present = True
        tmpid = tmpid[0][0]
        current_sat_file = sat_files[tmpid]
        
        ncfile = xarray.open_dataset(current_sat_file,decode_times=False)
        sat_time_epoch = np.array(ncfile['time_epoch'].copy())
        sat_lat = np.array(ncfile['lat'].copy())
        sat_lon = np.array(ncfile['lon'].copy())
        sat_visible_reflectance = np.array(ncfile['visible_reflectance'].copy())
        sat_ir_brightness_temperature = np.array(ncfile['ir_brightness_temperature'].copy())
        sat_effective_temperature = np.array(ncfile['effective_temperature'].copy())
        sat_lwp = np.array(ncfile['lwp'].copy())
        sat_iwp = np.array(ncfile['iwp'].copy())
        sat_ice_re = np.array(ncfile['ice_re'].copy())
        sat_ice_de = np.array(ncfile['ice_de'].copy())
        sat_liq_re = np.array(ncfile['liq_re'].copy())
        sat_optical_depth = np.array(ncfile['optical_depth'].copy())
        ncfile.close()
        print('Completed satellite processing.')                    

        
    #===========================================
    # End Satellite Block
    #===========================================    
    


    
    
    #===========================================
    # Begin miscellaneous block.
    #
    # Perform miscellaneous functions needed to
    # write output to netCDF files. Each sub-block
    # is defined below.
    #===========================================
    print('Begin cleanup for writing.')                    
    
    # make basta_time_ts array
    basta_time_ts = np.array([toTimestamp(basta_time_dt[dd]) for dd in range(len(basta_time_dt))])
    
    # if no data exist for both ceilometer, make a merged ceilometer flag
    if (ceilometer_present == False) and (aad_ceilometer_present == False):
        merged_ceilometer_present = False
    else:
        merged_ceilometer_present = True
        
    # if there is no ceilometer data present at all, then skip this date
    if merged_ceilometer_present == False:
        print('No ceilometer data avaialble. Skipping this date.')
        print('')
        ii+=1
        continue


    # Pull out 3 sounding times (if available) to be included in the output file.
    num_soundings = len(sonde_time_dt)
    target_time_1 = datetime.datetime(date.year,date.month,date.day,0)
    target_time_2 = datetime.datetime(date.year,date.month,date.day,12)
    tmp_time_delta = datetime.timedelta(days=1)
    target_time_3 = datetime.datetime(date.year,date.month,date.day,0) + tmp_time_delta
    
    sonde_time_dt = np.array(sonde_time_dt)
    
    tmpid = np.where((sonde_time_dt > (target_time_1 - datetime.timedelta(hours=2))) & (sonde_time_dt < (target_time_1 + datetime.timedelta(hours=2))))

    if np.size(tmpid) > 0:
        sounding_1_id = tmpid[0][0]
        sonde_1_flag = True
    else:
        sonde_1_flag = False
    tmpid = np.where((sonde_time_dt > (target_time_2 - datetime.timedelta(hours=2))) & (sonde_time_dt < (target_time_2 + datetime.timedelta(hours=2))))
    if np.size(tmpid) > 0:
        sounding_2_id = tmpid[0][0]  
        sonde_2_flag = True
    else:
        sonde_2_flag = False
    tmpid = np.where((sonde_time_dt > (target_time_3 - datetime.timedelta(hours=2))) & (sonde_time_dt < (target_time_3 + datetime.timedelta(hours=2))))
    if np.size(tmpid) > 0:
        sounding_3_id = tmpid[0][0]
        sonde_3_flag = True
    else:
        sonde_3_flag = False
        
        
    write_sonde_time_epoch = {}
    write_sonde_time_dt = {}
    write_sonde_temperature = {}
    write_sonde_rh = {}
    write_sonde_rh_i = {}
    write_sonde_pressure = {}
    write_sonde_q = {}
    write_sonde_height = {}
    write_sonde_u = {}
    write_sonde_v = {}
    write_sonde_wind_speed = {}
    write_sonde_wind_dir = {}
    write_sonde_theta = {}
    write_sonde_theta_e = {}
    write_sonde_time_dt_long = {}
    write_sonde_flag = {}
    
    sonde_flags = [sonde_1_flag,sonde_2_flag,sonde_3_flag]
    
    if sonde_1_flag == False:
        sounding_1_id = -999.
        write_sonde_flag['1'] = False
    else:
        write_sonde_flag['1'] = True
    if sonde_2_flag == False:
        sounding_2_id = -999.
        write_sonde_flag['2'] = False
    else:
        write_sonde_flag['2'] = True
    if sonde_3_flag == False:
        write_sonde_flag['3'] = False
        sounding_3_id = -999.
    else:
        write_sonde_flag['3'] = True
        
    sounding_ids = [sounding_1_id,sounding_2_id,sounding_3_id]
        
        
    
    for jj in range(len(sonde_flags)):
        if sonde_flags[jj] == True:
            # convert sonde_time t= epoch
            tmp_sonde_time_dt = sonde_time_dt[sounding_ids[jj]]
            tmp_sonde_time_epoch = toTimestamp(tmp_sonde_time_dt)
            write_sonde_time_epoch[str(int(jj+1))] = tmp_sonde_time_epoch
            write_sonde_time_dt[str(int(jj+1))] = tmp_sonde_time_dt
            write_sonde_temperature[str(int(jj+1))] = sonde_temperature[sounding_ids[jj]]
            write_sonde_pressure[str(int(jj+1))] = sonde_pressure[sounding_ids[jj]]
            write_sonde_rh[str(int(jj+1))] = sonde_rh[sounding_ids[jj]]
            write_sonde_rh_i[str(int(jj+1))] = sonde_rh_i[sounding_ids[jj]]
            tmp_sonde_height = sonde_height[sounding_ids[jj]]
            tmp_sonde_height = tmp_sonde_height*1.e3 # km to m
            write_sonde_height[str(int(jj+1))] = tmp_sonde_height
            write_sonde_u[str(int(jj+1))] = sonde_u[sounding_ids[jj]]
            write_sonde_v[str(int(jj+1))] = sonde_v[sounding_ids[jj]]
            write_sonde_wind_speed[str(int(jj+1))] = sonde_wind_speed[sounding_ids[jj]]
            write_sonde_wind_dir[str(int(jj+1))] = sonde_wind_dir[sounding_ids[jj]]
            write_sonde_q[str(int(jj+1))] = sonde_q[sounding_ids[jj]]
            write_sonde_theta[str(int(jj+1))] = sonde_theta[sounding_ids[jj]]
            write_sonde_theta_e[str(int(jj+1))] = sonde_theta_e[sounding_ids[jj]]
            write_sonde_time_dt_long[str(int(jj+1))] = sonde_time_dt_long[sounding_ids[jj]]
        else:
            write_sonde_time_epoch[str(int(jj+1))] = -999.
            write_sonde_time_dt[str(int(jj+1))] = -999.
            write_sonde_temperature[str(int(jj+1))] = -999.
            write_sonde_pressure[str(int(jj+1))] = -999.
            write_sonde_rh[str(int(jj+1))] = -999.
            write_sonde_rh_i[str(int(jj+1))] = -999.
            write_sonde_height[str(int(jj+1))] = -999.
            write_sonde_u[str(int(jj+1))] = -999.
            write_sonde_v[str(int(jj+1))] = -999.
            write_sonde_wind_speed[str(int(jj+1))] = -999.
            write_sonde_wind_dir[str(int(jj+1))] = -999.
            write_sonde_q[str(int(jj+1))] = -999.
            write_sonde_theta[str(int(jj+1))] = -999.
            write_sonde_theta_e[str(int(jj+1))] = -999.       
            write_sonde_time_dt_long[str(int(jj+1))] = -999.       
            

    
    if sfc_present == True:
        # Convert sfc met units
        sfc_temperature = sfc_temperature +273.15 # deg C to K
        sfc_temperature_interp = sfc_temperature_interp +273.15 # deg C to K
        sfc_pressure = sfc_pressure*10. # kPa to hPa
        sfc_pressure_interp = sfc_pressure_interp*10. # kPa to hPa

    print('Completed cleanup for writing.')                    
    
    #===========================================
    # End miscellaneous block.
    #===========================================     
    
    

    #===========================================
    # Begin dataset file writing.
    #===========================================
    out_dict = {'basta':{},\
                'ceil':{},\
                'sonde':{},\
                'sonde_native':{},\
                'sfc_met':{},\
                'cluster_id':{},\
                'sat':{},\
                'flags':{},\
               }
    # flags
    out_dict['flags'] = {'sfc_present':sfc_present,\
                         'sat_present':sat_present,\
                         'sonde_sufficient_mask':sonde_sufficient_mask}
    
    # radar
    out_dict['basta'] = {'ref':basta_ref,\
                         'vel':basta_vel,\
                         'time_dt':basta_time_dt,\
                         'bad_radar_data_flag':bad_radar_data_flag,\
                         'flag':basta_flag,\
                         'height':basta_height}
    
    # ceilometer
    out_dict['ceil'] = {'cbh':merged_ceil_cbh_1}
    
    # sonde
    out_dict['sonde'] = {'temperature':sonde_temperature_interp_2,\
                         'q':sonde_q_interp_2,\
                         'pressure':sonde_pressure_interp_2,\
                         'rh':sonde_rh_interp_2,\
                         'rh_i':sonde_rh_i_interp_2,\
                         'u':sonde_u_interp_2,\
                         'v':sonde_v_interp_2,\
                         'wind_speed':sonde_wind_speed_interp_2,\
                         'wind_dir':sonde_wind_dir_interp_2,\
                         'theta':sonde_theta_interp_2,\
                         'theta_e':sonde_theta_e_interp_2}

    
    # sonde_native
    out_dict['sonde_native'] = {'temperature':write_sonde_temperature,\
                         'q':write_sonde_q,\
                         'rh':write_sonde_rh,\
                         'rh_i':write_sonde_rh_i,\
                         'u':write_sonde_u,\
                         'v':write_sonde_v,\
                         'wind_speed':write_sonde_wind_speed,\
                         'wind_dir':write_sonde_wind_dir,\
                         'theta':write_sonde_theta,\
                         'theta_e':write_sonde_theta_e,\
                         'height':write_sonde_height,\
                         'pressure':write_sonde_pressure,\
                         'time_dt':write_sonde_time_dt,\
                         'time_dt_long':write_sonde_time_dt_long,\
                         'flag':write_sonde_flag,\
                        }    
  
    
    # sfc met
    if sfc_present == True:
        out_dict['sfc_met'] = {'temperature':sfc_temperature_interp,\
                               'rh':sfc_rh_interp,\
                               'wind_speed':sfc_wind_speed_interp,\
                               'wind_dir':sfc_wind_dir_interp,\
                               'pressure':sfc_pressure_interp,\
                               'wind_dir_native':sfc_wind_dir,\
                               'sfc_time_dt_native':sfc_time_dt,\
                              }

    # cluster id
    out_dict['cluster_id'] = {'cluster_id':cluster_id_interp}
    
    # sat
    if sat_present == True:
        out_dict['sat'] = {'lwp':sat_lwp,\
                           'iwp':sat_iwp,\
                           'ice_re':sat_ice_re,\
                           'ice_de':sat_ice_de,\
                           'liq_re':sat_liq_re,\
                           'tau':sat_optical_depth,\
                           'vis_ref':sat_visible_reflectance,\
                           'ir_tb':sat_ir_brightness_temperature,\
                           'eff_temp':sat_effective_temperature,\
                           'time_epoch':sat_time_epoch,\
                           'lat':sat_lat,\
                           'lon':sat_lon,\
                          }
    tmp_year = date.year
    tmp_month = date.month
    tmp_day = date.day
    save_path = '/mnt/raid/mwstanfo/merged_gridded/pkl/'
    STR = tmp_time.strftime("%Y_%m_%d")
    pkl_file = save_path+'micre_merged_gridded_instrument_{}.p'.format(STR)
    pickle.dump(out_dict,open(pkl_file,"wb"))     

    #print(aaaaaa)
    
    plt.contourf(basta_time_dt,basta_height,sonde_temperature_interp_2)
    print(aaaaa)
    print(' ')
   
    ii+=1



        

if False:    
    sonde_height_15_min = np.array(sonde_height_15_min)
    #print(np.shape(sonde_height_15_min))
    Fontsize=12
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.hist(sonde_height_15_min)
    ax.set_ylabel('Count',fontsize=Fontsize)
    ax.set_xlabel('Height of sonde within 30 min. [km]',fontsize=Fontsize)
    ax.tick_params(labelsize=Fontsize)
    ax.grid(which='both',c='grey')
    plt.show()
    plt.close()        
        
        