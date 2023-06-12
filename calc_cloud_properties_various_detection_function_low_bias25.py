#============================================================================
# calc_cloud_properties_various_detection_function.ipynb
#
# Calculates cloud properties using various D_min and Ze_min detection methods
# but low-biasing the CBH by 25 m (1 BASTA range gate).
#
# Author: McKenna W. Stanford
#============================================================================

#==================================================
# Imports
#==================================================
import numpy as np
import glob
import xarray
import datetime
import calendar
import pickle
import time
import os
from dask.distributed import Client, progress, LocalCluster
import dask
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator as nn
import matplotlib
import matplotlib.pyplot as plt



#===================================================
#===================================================
#===================================================
#=== MAIN FUNCTION TO CALCULATE CLOUD PROPERTIES ===
#===================================================
#===================================================
#===================================================
def calc_cloud_properties_low_bias25(infile,date,dmin,ze_thresh,**kwargs):
    #calc_normalized_props=True,calc_fog_props=True,calc_virga_props=True,arb_dbz_cutoff=False):
    calc_normalized_props = kwargs.get('calc_normalized_props',False)
    calc_virga_props = kwargs.get('calc_virga_props',False)
    calc_fog_props = kwargs.get('calc_fog_props',False)
    arb_dbz_cutoff = kwargs.get('arb_dbz_cutoff',False)
    #-------------------------------------------------------------
    # Load pickle file for specific date containing
    # the merged and gridded instruments. 
    #
    # All 2D variables are on the radar time and height
    # grid, all 1D-time variables are interpolated
    # to the radar time grid, and all 1D-height
    # variables are interpolated to the radar height
    # grid.
    #-------------------------------------------------------------
    print(infile)
    pkl_dict = pickle.load(open(infile,"rb"))
    print('Date:',date)
    #-------------------------------------------------------------
    # Initialize empty dictionary that will hold
    # the cloud properties for the entire date
    #-------------------------------------------------------------    
    props_dict = {}
    
    
    #-------------------------------------------------------------   
    # Get flags and masks necessary for calculations
    #-------------------------------------------------------------
    flags = pkl_dict['dataset_flags']
    sfc_present_flag = flags['sfc_met']
    interp_sonde_flag = pkl_dict['dataset_flags']['interp_sonde']
    merge_ceil_flag = pkl_dict['dataset_flags']['merge_ceil']
    #sat_present_flag = flags['sat_present']
    #sonde_sufficient_mask = flags['sonde_sufficient_mask']
    
    #-------------------------------------------------------------
    # SFC met flag, get SFC met variables    
    #-------------------------------------------------------------
    if sfc_present_flag:
        sfc_vars = pkl_dict['sfc_met']
        #sfc_temperature = sfc_vars['temperature']-273.15
        sfc_rh = sfc_vars['rh']
        #sfc_pressure = sfc_vars['pressure']
        #sfc_wind_speed = sfc_vars['wind_speed']
        #sfc_wind_dir = sfc_vars['wind_dir']
        #sfc_time_dt_native = sfc_vars['sfc_time_dt_native']

    #-------------------------------------------------------------
    # BASTA radar variables
    # Note that BASTA has already been QC'd here such
    # that any values lying below the minimum theoratical
    # reflectivity are set to -999. (i.e., hydrometeor free)
    # 
    # Also have a 'bad_radar_data_flag' that indicates
    # if data in *the entire* profile is bad.
    #
    # 'basta_flag' is simply the native QC flag
    #-------------------------------------------------------------
    basta_vars = pkl_dict['basta']
    ref = basta_vars['ref']
    vel = basta_vars['vel']
    time_dt = basta_vars['time_dt']
    bad_radar_data_flag = basta_vars['bad_radar_data_flag']
    height = basta_vars['height']
    basta_flag = basta_vars['flag']        
    
    #-------------------------------------------------------------
    # Height needs to be rounded to tenth decimal place due
    # to mismatches in significant digits (e.g., 1012.50006
    # needs to be rounded to 1012.5)
    #-------------------------------------------------------------
    height = np.around(height,2)

    #-------------------------------------------------------------
    # Merged ARM & U. of Canterbury ceilometer CBH
    #-------------------------------------------------------------
    # ***FROM CEIL METADATA***
    # Detection status is used for fog detection:
    # detection_status:flag_0_description = "No significant backscatter" ;
    # detection_status:flag_1_description = "One cloud base detected" ;
    # detection_status:flag_2_description = "Two cloud bases detected" ;
    # detection_status:flag_3_description = "Three cloud bases detected" ;
    # detection_status:flag_4_description = "Full obscuration determined but no cloud base detected" ;
    # detection_status:flag_5_description = "Some obscuration detected but determined to be transparent" ;
    # The detection status is time-interpolated and is the same for both ceilometers, though
    # there are no instancnes of flag_5 in the U. of Canterbury data--likely just due to post-processing.
    if merge_ceil_flag:
        ceil_cbh_1 = pkl_dict['merge_ceil']['cbh_1']  
        ceil_cbh_2 = pkl_dict['merge_ceil']['cbh_2']  
        ceil_cbh_3 = pkl_dict['merge_ceil']['cbh_3']  
        ceil_detection_status = pkl_dict['merge_ceil']['detection_status']
        
        thresh_cbh = 7537.5
        dumid = np.where(ceil_cbh_1 > thresh_cbh)
        if np.size(dumid) > 0.:
            ceil_cbh_1[dumid] = np.nan
        dumid = np.where(ceil_cbh_2 > thresh_cbh)
        if np.size(dumid) > 0.:
            ceil_cbh_2[dumid] = np.nan
        dumid = np.where(ceil_cbh_3 > thresh_cbh)
        if np.size(dumid) > 0.:
            ceil_cbh_3[dumid] = np.nan        
        
    else:
        print('No Ceilometer data on this day. Skipping this date.')
        return

    #------------------------------------------------------------- 
    # Synoptic Cluster ID
    # From Truong et al. (2020), JGRA,  https://doi.org/10.1029/2020JD033214
    # Cluster IDs are constructed using soundings, so "interpolated"
    # cluster IDs are taken to be valid within 6 hrs on either
    # side of a sounding. For any data not present within 6 hrs of a
    # sounding, the cluster ID is set to NaN.
    #
    # The cluster IDs, as integers, are as follows:
    #
    # -- 1: W1 (Poleward warm advection far ahead of cold front)
    # -- 2: M1 (High-pressure [large-scale subsidence])
    # -- 3: M2 (Cold front)
    # -- 4: M3 (Postfrontal)
    # -- 5: M4 (Warm front)
    # -- 6: C1 (Cicinity of cyclone center)
    #-------------------------------------------------------------
    #cluster_id = pkl_dict['cluster_id']['cluster_id']
    
    #-------------------------------------------------------------
    # Interpolated sounding for temperature
    #-------------------------------------------------------------
    if interp_sonde_flag:
        sounding_temperature = pkl_dict['interp_sonde']['temperature']-273.15
        seconds_to_nearest_sounding = pkl_dict['interp_sonde']['seconds_to_nearest_sounding'] # in timestamp
    
    
        # make sonde_sufficient_mask according to seconds to nearest_sounding
        sonde_sufficient_mask = np.zeros(np.shape(ceil_cbh_1))
        dum_thresh = 6*60.*60. # 6 hrs
        dumid = np.where(seconds_to_nearest_sounding <= dum_thresh)
        sonde_sufficient_mask[dumid] = 1
    
    #-----------------------------------------------
    # If all times have insufficient sonde data,
    # then skip this date entirely
    #-----------------------------------------------
    if not interp_sonde_flag:
        print('No Sonde data. Skipping this date')
        return
    if interp_sonde_flag:
        if np.all(sonde_sufficient_mask == 0.):
            print('Insufficient sonde data on this date. Skipping this date.')
            return

    #-----------------------------------------------
    # Grab 'dmin' (distance blow cloud base used for
    # precip detection) and ze_thresh (minimum
    # reflectivity threshold) and allocate them as
    # initial keys to 'props_dict'
    #-----------------------------------------------
    dmin_str = str(dmin)
    props_dict[dmin_str] = {}
    print('D_min:',dmin)
    
    ze_thresh_str = str(ze_thresh)  
    print('Z_e thresh:',ze_thresh)
    
    #-----------------------------------------------
    # Allocate props_dict with variables we will be computing
    # The variables are as follows:
    # --- 
    # --- time: 12-second time step as a datetime object
    # -------------------------------------------------
    # --- CLOUD GEOMETRIC & MACROPHYSICAL PROPERTIES
    # -------------------------------------------------
    # --- ctt: cloud top temperature (deg C)
    # --- cbt: cloud base temperature (deg C)
    # --- cth: cloud top height (m)
    # --- cbh: cloud base height (m)
    # --- c_thick: cloud layer thickness/depth (m)
    # -------------------------------------------------
    # --- PRECIPITATION PROPERTIES
    # -------------------------------------------------
    # --- IWflx: ice water flux (mm/hr)
    # --- LWflx: liquid water flux (mm/hr)
    # --- rcb: precipitation rate (IWflx + LWflx) (mm/hr)
    # --- sfc_IWflx: surface ice water flux (mm/hr)
    # --- sfc_LWflx: surface liquid water flux (mm/hr)
    # --- sfc_rcb: surface precipitaion rate (sfc_IWflx + sfc_LWflx) (mm/hr)
    # -------------------------------------------------
    # --- PRECIPITATION LAYER PROPERTIES
    # -------------------------------------------------
    # --- ref_mean_bel: layer-avg. reflectivity below CBH within dmin (dBZ)
    # --- vel_min_bel: layer-min. Dop. velocity below CBH within dmin (m/s)
    # --- T_mean_bel: layer-avg. temperature below CBH within dmin (deg C)
    # --- Z_min_mean_bel: layer-avg. minimum reflectivity below CBH within dmin (dBZ)
    # -------------------------------------------------
    # --- SFC PRECIPITATION LAYER PROPERTIES
    # -------------------------------------------------
    # --- sfc_ref_mean: layer-avg. reflectivity at hmin+dmin (dBZ)
    # --- sfc_vel_min: layer-min. Dop. velocity at hmin+dmin (m/s)
    # --- sfc_T_mean: layer-avg. temperature at hmin+dmin (deg C)
    # --- sfc_Z_min_mean: layer-avg. minimum reflectivity at hmin+dmin (dBZ)
    # -------------------------------------------------
    # --- BINARIES
    # -------------------------------------------------
    # --- cloud_binary: 0=no cloud; 1= cloud
    # --- precip_binary: 0=no precipitating cloud; 1=precipitation cloud
    # --- sfc_precip_binary: 0=no precip at sfc; 1=precip at sfc
    # --- precip_to_sfc_binary: 0=precipitating cloud doesn't reach sfc; 1 = precipitating cloud reaches sfc
    # --- precip_updraft_binary: 0=precip within neg. Dop. velocity; 1=precip within pos. Dop. velocity
    # --- sfc_precip_updraft_binary: 0=sfc precip within neg. Dop velocity; 1=sfc precip within pos. Dop. velocity
    # --- virga_binary: 0=no virga; 1=virga
    # --- fog_binary: 0=no fog; 1 = fog
    # --- fog_possible_binary: 0=no potential fog; 1=potential for fog
    # --- single_layer_binary: 0=only 1 cloud layer; 1 = multiple cloud layers
    # --- ref_at_cbh_binary: 0=no reflectivity at cloud base; 1=valid reflectivity at cloud base 
    # --- ref_within_100m_above_cbh_binary: 0=no valid reflectivity within 100m above CBH; 1=valid reflectivity within 100m above CBH
    # --- ref_within_100m_below_cbh_binary: 0=no valid reflectivity within 100m below CBH; 1=valid reflectivity within 100m below CBH
    # --- ref_above_cbh_binary: 0=no valid reflectivity in profile above CBH; 1=valid reflectivity in profile above CBH
    # --- ref_below_cbh_binary: 0=no valid reflectivity in profile below CBH; 1=valid reflectivity in profile below CBH
    # --- cloud_50m_thick_binary: 0=cloud thickness is less than 50m; 1=cloud thickness is greater than or equal to 50m
    # -------------------------------------------------
    # --- MISCELLANEOUS CLOUD STATISTICS
    # -------------------------------------------------
    # --- num_ref_cloud_layers_in_profile: number of cloud layers in profile determined by reflectivity
    # --- num_ceil_cbh_in_profile: number of cloud layers in profile determined by ceilometer (max 3)
    # --- cluster_id: synoptic cluster classification (integer)
    # --- nearest_ref_above_cbh: for ref_at_cbh_binary=0 and ref_above_cbh_binary=1, distance above CBH to nearest valid reflectivity value (m)
    # --- nearest_ref_below_cbh: for ref_at_cbh_binary=0 and ref_below_cbh_binary=1, distance below CBH to nearest valid reflectivity value (m)
    # -------------------------------------------------
    # --- SFC METEOROLOGY
    # -------------------------------------------------
    # --- sfc_rh: sfc met. relative humidity (%)
    # --- sfc_temperature: sfc met. temperature (deg C)
    # --- sfc_pressure: sfc met. pressure (hPa)
    # --- sfc_wind_speed: sfc met. wind speed (m/s)
    # --- sfc_wind_dir: sfc met. wind direction (deg)
    # -------------------------------------------------
    # --- MISCELLANEOUS CLOUD PROPERTIES
    # -------------------------------------------------   
    # --- virga_base_height: base of lowest valid, continuous reflectivity below cloud base (m)
    # --- native_cbh_1: merged ceilometer first CBH without processing
    # --- native_cbh_2: merged ceilometer second CBH without processing
    # --- native_cbh_3: merged ceilometer third CBH without processing
    # -------------------------------------------------
    # --- CBH-NORMALIZED PROFILE PROPERTIES
    # -------------------------------------------------    
    # --- ref_norm_prof_above: reflectivity profile above and including CBH (dBZ)
    # --- vel_norm_prof_above: Dop. velocity profile above and including CBH (m/s)
    # --- T_norm_prof_above: temperature profile above and including CBH (deg C)
    # --- height_norm_prof_above: height profile above and including CBH (m)
    # --- ref_norm_prof_bel: reflectivity profile below CBH (dBZ)
    # --- vel_norm_prof_bel: Dop. velocity profile below CBH (dBZ)
    # --- T_norm_prof_bel: temperature profile below CBH (deg C)
    # --- height_norm_prof_bel: height profile below CBH (m)
    # --- ref_norm_cbh: reflectivity profile normalized by CBH (dBZ)
    # --- vel_norm_cbh: Dop. velocity profile normalized by CBH (m/s)
    # --- height_norm_cbh: height profile normalized by CBH (m)
    # --- T_norm_cbh: emperature profile normalized by CBH (deg C)
    #-----------------------------------------------
    #prop_keys = ['time','ctt','cbt','cth','cbh','c_thick','IWflx','LWflx','sfc_IWflx','sfc_LWflx','rcb','sfc_rcb',\
    #            'ref_mean_bel','vel_min_bel','T_mean_bel','Z_min_mean_bel','sfc_ref_mean','sfc_vel_min','sfc_T_mean',\
    #            'sfc_Z_min_mean','cloud_binary','precip_binary','sfc_precip_binary','precip_to_sfc_binary','precip_updraft_binary',\
    #            'sfc_precip_updraft_binary','single_layer_binary','ref_at_cbh_binary','ref_within_100m_above_cbh_binary',\
    #            'ref_within_100m_below_cbh_binary','ref_above_cbh_binary','ref_below_cbh_binary','cloud_50m_thick_binary',\
    #            'num_ref_cloud_layers_in_profile','num_ceil_cbh_in_profile','cluster_id','nearest_ref_above_cbh',\
    #            'nearest_ref_below_cbh','sfc_rh','sfc_temperature','sfc_pressure','sfc_wind_speed',\
    #            'sfc_wind_dir','native_cbh_1','native_cbh_2','native_cbh_3','cbh_bel_min_binary']
    if False:
        prop_keys = ['time','ctt','cbt','cth','cbh','c_thick',\
                    'sfc_Z_min_mean','cloud_binary','precip_binary','sfc_precip_binary','precip_to_sfc_binary','precip_updraft_binary',\
                    'single_layer_binary','ref_at_cbh_binary','ref_within_100m_above_cbh_binary',\
                    'cloud_50m_thick_binary',\
                    'sfc_rh','cbh_bel_min_binary']
        props_dict[dmin_str][ze_thresh_str] = {}

        for prop_key in prop_keys:
            props_dict[dmin_str][ze_thresh_str][prop_key] = None

        # Fog property calculations are optional (user Boolean input)
        fog_props_keys = ['fog_binary','fog_possible_binary']
        if calc_fog_props:
            for fog_props_key in fog_props_keys:
                props_dict[dmin_str][ze_thresh_str][fog_props_key] = None

        # Virga property calculations are optional (user Boolean input)
        virga_props_keys = ['virga_binary','virga_base_height']
        if calc_virga_props:
            for virga_props_key in virga_props_keys:
                props_dict[dmin_str][ze_thresh_str][virga_props_key] = None

        # Normalized profile property calculations are optional (user Boolean input)
        normalized_props_keys = ['ref_norm_prof_above','vel_norm_prof_above','T_norm_prof_above','height_norm_prof_above',\
                                 'ref_norm_prof_bel','vel_norm_prof_bel','T_norm_prof_bel','height_norm_prof_bel',\
                                 'ref_norm_cbh','vel_norm_cbh','T_norm_cbh','height_norm_cbh']
        if calc_normalized_props:
            for normalized_props_key in normalized_props_keys:
                props_dict[dmin_str][ze_thresh_str][normalized_props_key] = None
        
    props_dict[dmin_str][ze_thresh_str] = {}

    #-----------------------------------------------
    #-----------------------------------------------
    #-----------------------------------------------
    #---------Calculate cloud properties------------
    #-----------------------------------------------
    #-----------------------------------------------
    #-----------------------------------------------
    
    
    #-------------------------------------------------    
    # Parameters
    #-------------------------------------------------    
    Radar_precip_depth_thresh = dmin # depth for precipitation calculations
    Radar_bin_width = 25. # BASTA vertical bin width
    Radar_precip_depth_bin_thresh = int(Radar_precip_depth_thresh/Radar_bin_width) # array length for depth of precip calculations
    
    
    num_times = len(time_dt) # number of 12-second time steps on date

    min_basta_loc = 6 #  -- lowest height index at which BASTA data is valid (i.e., QC'ing for surface clutter)
    hmin = height[min_basta_loc] # 162.5 m -- lowest height at which BASTA data is valid (i.e., QC'ing for surface clutter) # NOTE!: this is the midbin for the 150-175m bin edges
    min_cbh_thresh = hmin + dmin # lowest height at which precipitation calculations can be performed; requires that at least dmin depth be available below CBH for precip. detection

    ref_res = height[1]-height[0] # resolution of BASTA data (same as Radar_bin_width, 25)
    #min_c_thick = ref_res*2. # minimum cloud thickness
    min_sep_for_cat = 50. # minimum separation between valid reflectivity bins for concatenation of layers
    norm_prof_nz = 500 # number of indices used for normalized profiles, equates to 500 bins above or below CBH
    
    #-------------------------------------------------    
    # Calculate minimum detectable Ze threshold
    # as a function of height, where the input is
    # the minimum Ze at 1 km AGL and Z_min is calculated
    # following irradiance weakening inversely proportional
    # to the square of range. Takes 'ze_thresh' from
    # the function input as input
    #-------------------------------------------------
    Z_min_1km = ze_thresh # dBZ, input
    ref_range = 1000. # m
    Z_min = Z_min_1km + 20.*np.log10(height) - 20.*np.log10(ref_range) # dBZ profile
    Z_min[0] = np.nan
    
    
    #-------------------------------------------------
    # Boolean figure for plotting Z_min as a function
    # of height
    #-------------------------------------------------
    if False:
    #if True:
        fig = plt.figure(figsize=(6,6))
        Fontsize=14
        ax1 = fig.add_subplot(111)
        ax1.plot(Z_min,height*1.e-3,c='blue',lw=2,label='$Z_{e,min}$')
        ax1.set_ylabel('Height [km]',fontsize=Fontsize)
        ax1.set_xlabel('Z$_{e,min}$ [dBZ]',fontsize=Fontsize)
        ax1.tick_params(labelsize=Fontsize)
        tmpstr = 'Z$_{e}$ at 1 km: '+ze_thresh_str+' dBZ'
        ax1.text(0.1,0.9,tmpstr,transform=ax1.transAxes,fontsize=Fontsize*1.5)
        tmpstr = 'D$_{min}$: '+dmin_str+'m'
        ax1.text(0.1,0.8,tmpstr,transform=ax1.transAxes,fontsize=Fontsize*1.5)
        ax1.grid(which='both')
        ax1.axhline(1,c='navy',lw=3,label='1 km')
        ax1.legend(fontsize=Fontsize,loc='center left')
        ax1.set_title('$Z_{e,min}$ vs. Height',fontsize=Fontsize*1.5)
        plt.show()
        plt.close()
    
    

    
    #-------------------------------------------------
    # Initialize arrays that will hold cloud properties
    # and will be passed to the dictionary. These
    # will have the same names as the dictionary
    # keys. All values are initialized to -999.
    # Not all variables need to be initialized if they
    # are just pulled from what is read in from the pickle
    # file and are simply being passed to the cloud properties
    # dictionary (i.e., surface met, native ceilometer CBHs)
    #-------------------------------------------------

    ctt = np.zeros(num_times)-999.
    cbt = np.zeros(num_times)-999.
    cth = np.zeros(num_times)-999.
    cbh = np.zeros(num_times)-999.
    c_thick = np.zeros(num_times)-999.
    #IWflx = np.zeros(num_times)-999.
    #LWflx = np.zeros(num_times)-999.
    #sfc_LWflx = np.zeros(num_times)-999.
    #sfc_IWflx = np.zeros(num_times)-999.
    #rcb = np.zeros(num_times)-999.
    #sfc_rcb = np.zeros(num_times)-999.
    ref_mean_bel = np.zeros(num_times)-999.
    vel_min_bel = np.zeros(num_times)-999.
    T_mean_bel = np.zeros(num_times)-999.
    Z_min_mean_bel = np.zeros(num_times)-999.
    sfc_ref_mean = np.zeros(num_times)-999.
    sfc_vel_min = np.zeros(num_times)-999.
    sfc_T_mean = np.zeros(num_times)-999.
    sfc_Z_min_mean = np.zeros(num_times)-999.
    cloud_binary = np.zeros(num_times)-999.
    precip_binary = np.zeros(num_times)-999.
    sfc_precip_binary = np.zeros(num_times)-999.
    precip_to_sfc_binary = np.zeros(num_times)-999.
    precip_updraft_binary = np.zeros(num_times)-999.
    sfc_precip_updraft_binary = np.zeros(num_times)-999.
    if calc_virga_props:
        virga_base_height = np.zeros(num_times)-999.
        virga_binary = np.zeros(num_times)-999.
    if calc_fog_props:
        fog_binary = np.zeros(num_times)-999.
        fog_possible_binary = np.zeros(num_times)-999.
    single_layer_binary = np.zeros(num_times)-999.
    cbh_bel_min_binary = np.zeros(num_times)-999.
    ref_at_cbh_binary = np.zeros(num_times)-999.
    ref_within_100m_above_cbh_binary = np.zeros(num_times)-999.
    ref_within_100m_below_cbh_binary = np.zeros(num_times)-999.
    ref_above_cbh_binary = np.zeros(num_times)-999.
    ref_below_cbh_binary = np.zeros(num_times)-999.
    cloud_50m_thick_binary = np.zeros(num_times)-999.
    num_ref_cloud_layers_in_profile = np.zeros(num_times)-999.
    num_ceil_cbh_in_profile = np.zeros(num_times)-999.
    nearest_ref_above_cbh = np.zeros(num_times)-999.
    nearest_ref_below_cbh = np.zeros(num_times)-999.
    if calc_normalized_props:
        ref_norm_prof_above = np.zeros((num_times,norm_prof_nz))-999.
        vel_norm_prof_above = np.zeros((num_times,norm_prof_nz))-999.
        T_norm_prof_above = np.zeros((num_times,norm_prof_nz))-999.
        height_norm_prof_above = np.zeros((num_times,norm_prof_nz))-999.
        ref_norm_prof_bel = np.zeros((num_times,norm_prof_nz))-999.
        vel_norm_prof_bel = np.zeros((num_times,norm_prof_nz))-999.
        T_norm_prof_bel = np.zeros((num_times,norm_prof_nz))-999.
        height_norm_prof_bel = np.zeros((num_times,norm_prof_nz))-999.
        T_norm_cbh = np.zeros((num_times,norm_prof_nz))-999.
        ref_norm_cbh = np.zeros((num_times,norm_prof_nz))-999.
        vel_norm_cbh = np.zeros((num_times,norm_prof_nz))-999.
        height_norm_cbh = np.zeros((num_times,norm_prof_nz))-999.
    
    
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    # Loop through individual radar times
    # (12-second temporal resolution) over
    # and entire day
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    dumit=0 # dummy iteration used in dev. to isolate specific output
    
    for tt in range(num_times):
        
        # Skip time step if there is insufficient sonde data
        if sonde_sufficient_mask[tt] == 0.:
            continue

        # Skip time step if the radar data in profile is bad
        if bad_radar_data_flag[tt] == 1:
            continue
        
        # if there is no ceilometer data, skip time step
        #if np.isnan(ceil_cbh_1[tt]):
        #    continue
            
        # Limit variables to single times for easier workflow
        single_time_ref_prof = ref[:,tt]
        single_time_vel_prof = vel[:,tt]
        single_time_cbh = ceil_cbh_1[tt]
        single_time_T_prof = sounding_temperature[:,tt]
        single_time_ceil_ds = ceil_detection_status[tt]
        if sfc_present_flag > 0.:
            single_time_sfc_rh = sfc_rh[tt]
        
        #print('single_time_ref_prof type:',type(single_time_ref_prof))
        #print('single_time_vel_prof type:',type(single_time_vel_prof))
        #print('single_time_T_prof type:',type(single_time_T_prof))
        #print('single_time_cbh:',type(single_time_cbh))
        #print('single_time_ceil_ds type:',type(single_time_ceil_ds))
        #print('single_time_sfc_rh type:',type(single_time_sfc_rh))
        
        #print('single_time_ref_prof shape:',np.shape(single_time_ref_prof))
        #print('single_time_vel_prof shape:',np.shape(single_time_vel_prof))
        #print('single_time_T_prof shape:',np.shape(single_time_T_prof))
        #print('single_time_cbh:',single_time_cbh)
        #print('single_time_ceil_ds:',single_time_ceil_ds)
        #print('single_time_sfc_rh:',single_time_sfc_rh)

        
        # Convert from dBZ to equivalent reflectivity (Ze)
        valid_id = np.where((~np.isnan(single_time_ref_prof) | (single_time_ref_prof > -999.)))
        if np.size(valid_id) > 0.:
            single_time_ze_prof = np.zeros(np.shape(single_time_ref_prof))
            single_time_ze_prof[valid_id] = 10.**(single_time_ref_prof[valid_id]/10.)
        invalid_id = np.where(single_time_ref_prof == -999.)
        if np.size(invalid_id) > 0.:
            single_time_ze_prof[invalid_id] = 0.
        nanid = np.where(np.isnan(single_time_ref_prof))
        if np.size(nanid) > 0.:
            single_time_ze_prof[nanid] = np.nan
            
        
        
        #-------------------------------------------------
        #-------------------------------------------------
        #-------------------------------------------------
        # STEP 1: Check to see if a valid CBH (i.e., not NaN)
        # is present. If not, there is considered no cloud (cloud_binary==0).
        #
        # However, we look at the ceilometer detection status
        # (single_time_ceil_ds). If single_time_ceil_ds==4 (Full 
        # obscuration determined but no cloud base detected), there 
        # is the potential that this is fog.
        #
        # Then, if there is valid sfc meteorology data, check
        # the SFC RH. If sfc_rh >= 95 %, then set fog_binary==1.
        #
        # If sfc_rh < 95%, but single_time_ceil_ds==4, then
        # it is still possible that fog is present (since
        # we don't know the uncertainty of the sfc hygrometer
        # nor the optimal threhsold in sfc_rh for detecting
        # fog). In this case, set fog_binary==0 but set
        # fog_possible_binary==1.
        #
        # However, if single_time_cbh==NaN and single_time_ceil_ds
        # is not equal to 4, then there is either no
        # backscatter or "Some obscuration detected but 
        # determined to be transparent
        #
        # If there is no sfc met. data, then fog_binary
        # and fog_possible_binary are set to -555 (undetermined).
        #
        # After this evaluation, continue to the next
        # time step without running the remainder of the
        # algorithm.
        #-------------------------------------------------        
        #-------------------------------------------------        
        #-------------------------------------------------        
        if np.isnan(single_time_cbh):
            cloud_binary[tt] = 0
            cbh[tt] = np.nan
            if calc_fog_props:
                if sfc_present_flag > 0.:
                    if (single_time_ceil_ds == 4.) & (single_time_sfc_rh >= 95):
                        fog_binary[tt] = 1
                        fog_possible_binary[tt] = 1
                    elif (single_time_ceil_ds == 4.) & (single_time_sfc_rh < 95):
                        fog_possible_binary[tt] = 1
                        fog_binary[tt] = 0.
                    else:
                        fog_possible_binary[tt] = 0
                        fog_binary[tt] = 0
                else:
                    fog_binary[tt] = -555.
                    fog_possible_binary[tt] = -555.
                
                
            # For reporting purposes, we may be interested in how
            # often precip is identified at the surface but without
            # an overlying CBH detected by the ceilometer. This would
            # only happen if CBH is NaN but the detection status
            # is not 1-4. It may happen for detection status 0
            # (no significant backscatter) or 5 (some obscuration
            # but transparent). However, this seems unlikely, so
            # putting a pin in it for now.
        
            continue

        #-------------------------------------------------
        #-------------------------------------------------
        #-------------------------------------------------
        # STEP 2: If single_time_cbh is not NaN and thus
        # a CBH is detected.
        #-------------------------------------------------        
        #-------------------------------------------------        
        #-------------------------------------------------  

        elif ~np.isnan(single_time_cbh):   
            #print(single_time_cbh)
            # Low bias the CBH by 1 bin
            single_time_cbh = single_time_cbh - 25.
            #print(single_time_cbh)
            #print(aaaa)

            
            
            
            #-------------------------------------------------
            #-------------------------------------------------
            #-------------------------------------------------
            # STEP 2.1: If CBH is lower than the minimum height
            # possible for precip detection (min_cbh_thresh = 
            # hmin + dmin), no precipitation detection is
            # possible.
            #
            # In these cases, we will again check for fog
            # following Step 1. Here though, the CBH must
            # be less than 250 m, so this will only be valid
            # for hmin=150(/137.5) + dmin=100 and smaller.
            #
            # For continuity purposes, we will set cloud_binary
            # to zero for these cases.
            #-------------------------------------------------        
            #-------------------------------------------------        
            #------------------------------------------------- 
            if single_time_cbh < min_cbh_thresh:
                cbh_bel_min_binary[tt] = 1
                cloud_binary[tt] = 0.
                cbh[tt] = single_time_cbh
                if calc_fog_props:
                    if sfc_present_flag > 0.:
                        if single_time_cbh < 250.:
                            if single_time_sfc_rh >= 95:
                                fog_binary[tt] = 1
                                fog_possible_binary[tt] = 1
                            elif single_time_sfc_rh < 95:
                                fog_binary[tt] = 0
                                fog_possible_binary[tt] = 1
                        else:
                            fog_binary[tt] = 0
                            fog_possible_binary[tt] = 0
                    else:
                        fog_binary[tt] = -555.
                        fog_possible_binary[tt] = -555.
                continue    
                
            #-------------------------------------------------
            #-------------------------------------------------
            #-------------------------------------------------
            # STEP 2.2: If CBH is higher than the minimum height
            # possible for precip detection (min_cbh_thresh = 
            # hmin + dmin), no precipitation detection is
            # possible.
            #
            # Here we'll also check for fog if the CBH is less
            # than 250 m. Note that it is still possible for
            # this cloud to be precipitating, so we'll still
            # calculate precipitation statistics. For continuity,
            # we'll still mark fog_binary == 1 if all conditions
            # are met. In further processing, the cbh_bel_min_binary
            # Boolean will be used to isolate these specific
            # cases.
            #-------------------------------------------------        
            #-------------------------------------------------        
            #-------------------------------------------------  

            elif (single_time_cbh >= min_cbh_thresh):
                
                # Fog check
                if calc_fog_props:
                    if sfc_present_flag > 0.:
                        if single_time_cbh < 250.:
                            if single_time_sfc_rh >= 95:
                                fog_binary[tt] = 1
                                fog_possible_binary[tt] = 1
                            elif single_time_sfc_rh < 95:
                                fog_binary[tt] = 0
                                fog_possible_binary[tt] = 1
                        else:
                            fog_binary[tt] = 0
                            fog_possible_binary[tt] = 0
                    else:
                        fog_binary[tt] = -555.
                        fog_possible_binary[tt] = -555.
                    
                
                #-------------------------------------------------           
                # There is a cloud here, able to be identified
                # as precipitating or not, and the CBH is above
                # the min_cbh_thresh
                #-------------------------------------------------  
                cbh_bel_min_binary[tt] = 0
                cloud_binary[tt] = 1
                
                # Find the height ID matching the CBH
                height_id = np.where(height == single_time_cbh)[0][0]

                #-------------------------------------------------  
                # Normalize profiles by CBH, both above and below
                #-------------------------------------------------  
                # Above CBH -- includes CBH bin reflectivity value
                ref_above_cbh_prof = single_time_ref_prof[height_id:]
                vel_above_cbh_prof = single_time_vel_prof[height_id:]
                T_above_cbh_prof = single_time_T_prof[height_id:]
                height_above_cbh_prof = height[height_id:]

                # Below CBH -- last index is one bin below cloud base
                ref_below_cbh_prof = single_time_ref_prof[0:height_id]
                vel_below_cbh_prof = single_time_vel_prof[0:height_id]
                T_below_cbh_prof = single_time_T_prof[0:height_id]
                height_below_cbh_prof = height[0:height_id]
                Z_min_below_cbh_prof = Z_min[0:height_id]
                

                #-------------------------------------------------                 
                #-------------------------------------------------                 
                # STEP 2.2.1: There is not valid reflectivity at 
                # cloud base (i.e., the reflectivity at cloud
                # base is NaN [bad data] or -999. [no hydrometeor
                # signal]).
                #
                # There are several properties from this profile
                # that we will still calculate to investigate
                # these cases further.
                #------------------------------------------------- 
                #------------------------------------------------- 
                if (np.isnan(ref_above_cbh_prof[0])) or (ref_above_cbh_prof[0] == -999.):

                    ref_at_cbh_binary[tt] = 0
                    
                    #------------------------------------------------- 
                    # Check to see if there is any valid reflectivity
                    # below CBH.
                    # This is reported using 'ref_below_cbh_binary'
                    # This is done here only to see how often there
                    # is an altitude difference between the ceilometer
                    # CBH and the first valid reflectivity below
                    # cloud base.
                    #-------------------------------------------------
                    ref_below_id = np.where( (ref_below_cbh_prof > -999.) & (~np.isnan(ref_below_cbh_prof)))[0]
                    if np.size(ref_below_id) > 0.:
                        ref_below_cbh_binary[tt] = 1
                    
                        #------------------------------------------------- 
                        # Find the nearest reflectivity below cloud base
                        # and record the altitude difference.
                        #-------------------------------------------------
                        nearest_ref_below_cbh_height = height_below_cbh_prof[ref_below_id[-1]]
                        ref_below_alt_diff = single_time_cbh - nearest_ref_below_cbh_height
                        nearest_ref_below_cbh[tt] = ref_below_alt_diff
                        if ref_below_alt_diff <= 100.:
                            ref_within_100m_below_cbh_binary[tt] = 1
                        else:
                            ref_within_100m_below_cbh_binary[tt] = 0
                        
                    else:
                        ref_below_cbh_binary[tt] = 0
                        
                    
                    #------------------------------------------------- 
                    # Check to see if there is any valid reflectivity
                    # above CBH.
                    # This is reported using 'ref_above_cbh_binary'
                    #------------------------------------------------- 
                    ref_above_id = np.where((ref_above_cbh_prof > -999.) & (~np.isnan(ref_above_cbh_prof)))[0]
                    if np.size(ref_above_id) > 0.:
                        ref_above_cbh_binary[tt] = 1
                        
                        #------------------------------------------------- 
                        # Find the nearest reflectivity above cloud base
                        # and record the altitude difference.
                        #-------------------------------------------------
                        nearest_ref_above_cbh_height = height_above_cbh_prof[ref_above_id[0]]
                        ref_above_alt_diff = nearest_ref_above_cbh_height - single_time_cbh
                        nearest_ref_above_cbh[tt] = ref_above_alt_diff
                        
                        #------------------------------------------------- 
                        # Check to see if the nearest reflectivity value
                        # above cloud base is less than or equal to 100 m
                        # from CBH.
                        #-------------------------------------------------                       
                        
                        if ref_above_alt_diff <= 100.:
                            ref_within_100m_above_cbh_binary[tt] = 1
                            
                            #------------------------------------------------- 
                            # Valid reflectivity exists within 100 m above CBH.
                            #
                            # Create a cloud layer mask where CBH == 1. Then,
                            # find contiguous reflectivity objects above CBH.
                            # Some values within this 100 m depth above CBH
                            # may have invalid reflectivity. We will find the
                            # first valid reflectivity within 100 m above CBH
                            # and flag the invalid values *in between* as 
                            # cloud, and then will use the contiguous objects
                            # to find the cloud layer depth and other base/top
                            # properties. We will also allow for concatenation
                            # if the objects are separated by a minimum
                            # distance set by "min_sep_for_cat", for which
                            # the default value is 50 m.
                            #
                            # Note: Physically, this should correspond to
                            # situtations where the ceilometer's higher
                            # sensitivity receives backscatter from very small
                            # cloud droplets below cloud base that are not
                            # yet identifiable via the radar's sensitivity.
                            #-------------------------------------------------                               
                            cloud_layer_mask = np.zeros(len(ref_above_cbh_prof))
                            
                            # Mark CBH as a "1" in cloud_layer_mask
                            cloud_layer_mask[0] = 1
                            
                            # Now identify cloudy grid cells using reflectivity
                            valid_ref_id = np.where(ref_above_cbh_prof > -999.)
                            if np.size(valid_ref_id) > 0.:
                                cloud_layer_mask[valid_ref_id] = 1

                            # Identify contiguous cloudy grid cells
                            cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)

                            # If there is more than one cloud object, concatenate cloud layers
                            # separated by a minimum of min_sep_for_cat (default = 50 m)
                            if num_cloud_objects > 1:
                                alt_diff = np.zeros(num_cloud_objects-1)

                                unique_ids = np.unique(cloud_Objects)
                                unique_ids = unique_ids[1:]

                                for jj in range(len(unique_ids)-1):
                                    ids_layer_1 = np.where(cloud_Objects == jj+1)
                                    ids_layer_1 = ids_layer_1[0]
                                    ids_layer_2 = np.where(cloud_Objects == jj+2)
                                    ids_layer_2 = ids_layer_2[0]

                                    if np.size(ids_layer_1) == 1:
                                        layer_1_top = ids_layer_1[0]
                                    else:
                                        layer_1_top = ids_layer_1[-1]

                                    if np.size(ids_layer_2) == 1:
                                        layer_2_bottom = ids_layer_2[0]
                                    else:
                                        layer_2_bottom = ids_layer_2[0]

                                    alt_diff[jj] =  (layer_2_bottom-layer_1_top)*ref_res     


                                # create a new cloud mask that accounts for concatenating layers
                                # separated by less than or equal to 50 m
                                for jj in range(len(alt_diff)):
                                    if alt_diff[jj] <= (min_sep_for_cat):
                                        ids_layer_1 = np.where(cloud_Objects == jj+1)
                                        ids_layer_1 = ids_layer_1[0]
                                        ids_layer_2 = np.where(cloud_Objects == jj+2)
                                        ids_layer_2 = ids_layer_2[0]
                                        cloud_layer_mask[ids_layer_1[0]:ids_layer_2[-1]+1] = 1
                                        
                                # Redo object identification
                                cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)   

                            #-------------------------------------------------                               
                            # Boolean plot to check this specific case of cloud layer ID
                            #-------------------------------------------------
                            if False:
                                Fontsize=14
                                fig = plt.figure(figsize=(8,6))
                                ax1=fig.add_subplot(111)
                                ax1.set_ylabel('Height [m]',fontsize=Fontsize)
                                ax1.set_xlabel('Reflectivity [dBZ]',fontsize=Fontsize)
                                ax1.plot(single_time_ref_prof,height,lw=2,c='blue',label='Reflectivity')
                                ax1.axhline(single_time_cbh,c='navy',lw=3)
                                ax1.set_xlim(-65,20)
                                ax1.tick_params(labelsize=Fontsize)
                                ax1.grid(which='both')
                                if num_cloud_objects > 0.:
                                    dumx=ax1.get_xlim()
                                    dumx = np.arange(dumx[0],dumx[1]+1)
                                    for kk in range(num_cloud_objects):
                                        dumxid = np.where(cloud_Objects == kk+1)
                                        dumxid = np.squeeze(dumxid)
                                        if np.size(dumxid) > 1.:
                                            if kk == 0.:
                                                ax1.fill_between(dumx,height_above_cbh_prof[dumxid[0]],height_above_cbh_prof[dumxid[-1]],facecolor='lightgrey',label='cloud mask')
                                            else:
                                                ax1.fill_between(dumx,height_above_cbh_prof[dumxid[0]],height_above_cbh_prof[dumxid[-1]],facecolor='lightgrey')
                                        else:
                                            ax1.axhline(height_above_cbh_prof[dumxid],c='darkorange',lw=2,ls='dotted',label='single height valid ref')
                                            
                                dumxid = np.where(cloud_Objects == 1)
                                dumxid = np.squeeze(dumxid)
                                if np.size(dumxid) > 1:
                                    max_height_id = height_above_cbh_prof[dumxid[-1]]+100.
                                    min_height_id = single_time_cbh-100.
                                    if min_height_id < 0.:
                                        min_height_id = 0.
                                else:
                                    max_height_id = height_above_cbh_prof[dumxid]+100.
                                    min_height_id = single_time_cbh-100.
                                    if min_height_id < 0.:
                                        min_height_id = 0.
                                ax1.legend(fontsize=Fontsize,bbox_to_anchor=(0.5,-0.25),loc='lower center',ncol=3)
                                
                                ax1.set_ylim(min_height_id,max_height_id)
                                dumid = np.where(cloud_Objects == 1)[0]
                                tmp_c_thick = (len(dumid))*ref_res
                                ax1.set_title('1$^{st}$ Cloud Layer Thickness = '+str(tmp_c_thick)+'m',fontsize=Fontsize*1.5)
                                plt.show()
                                plt.close()                           
                            
                            #-------------------------------------------------                               
                            # Calculate macrophysical and base/height properties
                            #-------------------------------------------------                            

                            first_cloud_object_mask_id = np.where(cloud_Objects == 1)[0]
                            tmp_c_thick = (len(first_cloud_object_mask_id))*ref_res    
                            c_thick[tt] = tmp_c_thick
                            
                            if tmp_c_thick >= 50:
                                cloud_50m_thick_binary[tt] = 1
                            else:
                                cloud_50m_thick_binary[tt] = 0

                            cbt[tt] = T_above_cbh_prof[first_cloud_object_mask_id[0]]
                            ctt[tt] = T_above_cbh_prof[first_cloud_object_mask_id[-1]]
                            cth[tt] = height_above_cbh_prof[first_cloud_object_mask_id[-1]]
                            cbh[tt] = single_time_cbh                                    

                        else:
                            #------------------------------------------------- 
                            # Valid reflectivity does not exist within 100 m
                            # above cloud base. 
                            # In this case, the cloud top height and cloud top
                            # temperature will be the same as the cloud base
                            # height and temperature and the cloud thickness will
                            # be set to 0 m. 
                            # record the cloud base temperature and height,
                            # but thickness and cloud top properties are
                            # not compuated.
                            #-------------------------------------------------                                    
                            ref_within_100m_above_cbh_binary[tt] = 0
                            cbt[tt] = T_above_cbh_prof[0]
                            ctt[tt] = T_above_cbh_prof[0]
                            cbh[tt] = single_time_cbh
                            cth[tt] = single_time_cbh
                            c_thick[tt] = 0.
                            cloud_50m_thick_binary[tt] = 0

                            
                            # To identify single layers, we will want to see if there
                            # are any overlying cloud layers recognizable via reflectivity.
                            # The rest of this conditional block is therefore only to 
                            # get "num_cloud_objects"
                            
                            # Mask reflectivity
                            cloud_layer_mask = np.zeros(len(ref_above_cbh_prof))
                            
                            # Make the first index a cloud because of the CBH recognition
                            # by the ceilometer
                            cloud_layer_mask[0] = 1
                            
                            valid_ref_id = np.where(ref_above_cbh_prof > -999.)
                            if np.size(valid_ref_id) > 0.:
                                cloud_layer_mask[valid_ref_id] = 1

                            # Identify contiguous reflectivity objects
                            cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)

                            # Concatenate cloud layers
                            if num_cloud_objects > 1:
                                alt_diff = np.zeros(num_cloud_objects-1)

                                unique_ids = np.unique(cloud_Objects)
                                unique_ids = unique_ids[1:]

                                for jj in range(len(unique_ids)-1):
                                    ids_layer_1 = np.where(cloud_Objects == jj+1)
                                    ids_layer_1 = ids_layer_1[0]
                                    ids_layer_2 = np.where(cloud_Objects == jj+2)
                                    ids_layer_2 = ids_layer_2[0]

                                    if np.size(ids_layer_1) == 1:
                                        layer_1_top = ids_layer_1[0]
                                    else:
                                        layer_1_top = ids_layer_1[-1]

                                    if np.size(ids_layer_2) == 1:
                                        layer_2_bottom = ids_layer_2[0]
                                    else:
                                        layer_2_bottom = ids_layer_2[0]

                                    alt_diff[jj] =  (layer_2_bottom-layer_1_top)*ref_res    

                                # create a new cloud mask that accounts for concatenating layers
                                for jj in range(len(alt_diff)):
                                    if alt_diff[jj] <= (min_sep_for_cat):
                                        ids_layer_1 = np.where(cloud_Objects == jj+1)
                                        ids_layer_1 = ids_layer_1[0]
                                        ids_layer_2 = np.where(cloud_Objects == jj+2)
                                        ids_layer_2 = ids_layer_2[0]
                                        cloud_layer_mask[ids_layer_1[0]:ids_layer_2[-1]+1] = 1
                                # Redo object identification
                                cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)

                    else:
                        #------------------------------------------------- 
                        # No valid reflectivity at all in profile
                        # above CBH. Cloud top height and temperature
                        # will be set to the base height and temperature
                        #-------------------------------------------------                              
                        #ref_above_cbh_binary[tt] = 0
                        cbt[tt] = T_above_cbh_prof[0]
                        ctt[tt] = T_above_cbh_prof[0]
                        cbh[tt] = single_time_cbh
                        cth[tt] = single_time_cbh
                        c_thick[tt] = 0.
                        cloud_50m_thick_binary[tt] = 0
                        num_cloud_objects = 1
                        ref_within_100m_above_cbh_binary[tt] = 0
                
                else:
                    #-------------------------------------------------                 
                    #-------------------------------------------------                 
                    # STEP 2.2.2: There *is* valid reflectivity at 
                    # cloud base
                    #-------------------------------------------------                     
                    #-------------------------------------------------                     
                    ref_at_cbh_binary[tt] = 1
                    ref_above_cbh_binary[tt] = 1
                    
                    # Mask reflectivity

                    cloud_layer_mask = np.zeros(len(ref_above_cbh_prof))
                    valid_ref_id = np.where(ref_above_cbh_prof > -999.)
                    if np.size(valid_ref_id) > 0.:
                        cloud_layer_mask[valid_ref_id] = 1
                        
                    # Create contiguous reflectivity objects
                    cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)

                    # Concatenate cloud layers
                    if num_cloud_objects > 1:
                        alt_diff = np.zeros(num_cloud_objects-1)

                        unique_ids = np.unique(cloud_Objects)
                        unique_ids = unique_ids[1:]

                        for jj in range(len(unique_ids)-1):
                            ids_layer_1 = np.where(cloud_Objects == jj+1)
                            ids_layer_1 = ids_layer_1[0]
                            ids_layer_2 = np.where(cloud_Objects == jj+2)
                            ids_layer_2 = ids_layer_2[0]

                            if np.size(ids_layer_1) == 1:
                                layer_1_top = ids_layer_1[0]
                            else:
                                layer_1_top = ids_layer_1[-1]

                            if np.size(ids_layer_2) == 1:
                                layer_2_bottom = ids_layer_2[0]
                            else:
                                layer_2_bottom = ids_layer_2[0]

                            alt_diff[jj] =  (layer_2_bottom-layer_1_top)*ref_res   



                        # Create a new cloud mask that accounts for concatenating layers
                        for jj in range(len(alt_diff)):
                            if alt_diff[jj] <= (min_sep_for_cat):
                                ids_layer_1 = np.where(cloud_Objects == jj+1)
                                ids_layer_1 = ids_layer_1[0]
                                ids_layer_2 = np.where(cloud_Objects == jj+2)
                                ids_layer_2 = ids_layer_2[0]                                
                                cloud_layer_mask[ids_layer_1[0]:ids_layer_2[-1]+1] = 1

                        # Redo object identification
                        cloud_Objects,num_cloud_objects = ndimage.label(cloud_layer_mask)
                    #-------------------------------------------------                               
                    # Calculate macrophysical and base/height properties
                    #-------------------------------------------------   
                    first_cloud_object_mask_id = np.where(cloud_Objects == 1)[0]
                    tmp_c_thick = (len(first_cloud_object_mask_id))*ref_res
                    c_thick[tt] = tmp_c_thick
                    
                    if tmp_c_thick >= 50:
                        cloud_50m_thick_binary[tt] = 1
                    else:
                        cloud_50m_thick_binary[tt] = 0

                    c_thick[tt] = tmp_c_thick
                    cbt[tt] = T_above_cbh_prof[first_cloud_object_mask_id[0]]
                    ctt[tt] = T_above_cbh_prof[first_cloud_object_mask_id[-1]]
                    cth[tt] = height_above_cbh_prof[first_cloud_object_mask_id[-1]]
                    cbh[tt] = single_time_cbh   
                    
                #-------------------------------------------------                 
                #-------------------------------------------------                 
                # End STEP 2.2.1 (valid reflectivity at cloud base)
                #------------------------------------------------- 
                #------------------------------------------------- 
                
                #-------------------------------------------------
                #-------------------------------------------------
                #-------------------------------------------------
                # STEP 2.3: CBH is higher than the minimum height
                # possible for precip detection (min_cbh_thresh = 
                # hmin + dmin). Note that this STEP is technically
                # still inside Step 2.2 (conditional on CBH >
                # min_cbh_thresh and not NaN).
                #
                # Calculate precipitation rates and occurrence
                # frequency (i.e., binary)
                #
                # Also determine number of clouds in layer,
                # if only a single layer, mark single_layer_binary
                # == 1, and if the options are valid, calculate
                # virga properties and normalized profile properties.
                #
                # NOTE: The only requirement to check whether
                # or not a cloud layer is precipitating is that
                # valid cloud base height is detected. The thickness
                # nor the presence of valid coincident reflectivity
                # does not matter. 
                #-------------------------------------------------        
                #-------------------------------------------------        
                #-------------------------------------------------    


                # Assign properties according to number of layers
                if num_cloud_objects > 1:
                    single_layer_binary[tt] = 0
                else:
                    single_layer_binary[tt] = 1 

                num_ref_cloud_layers_in_profile[tt] = num_cloud_objects
                                               
                #-------------------------------------------------                    
                #-------------------------------------------------
                # Calculate precipitation statistics
                #-------------------------------------------------                    
                #-------------------------------------------------                    
                
                # Grab distance below CBH according to dmin
                ref_bel_cbh = single_time_ref_prof[height_id-Radar_precip_depth_bin_thresh:height_id]
                vel_bel_cbh = single_time_vel_prof[height_id-Radar_precip_depth_bin_thresh:height_id]
                T_bel_cbh = single_time_T_prof[height_id-Radar_precip_depth_bin_thresh:height_id]
                height_bel_cbh = height[height_id-Radar_precip_depth_bin_thresh:height_id]
                Z_min_bel_cbh = Z_min[height_id-Radar_precip_depth_bin_thresh:height_id]
                        
                #-------------------------------------------------
                # Find valid reflectivity below cloud base
                # according to dmin and determine if the values
                # are greater than the minimum detectable Ze
                #-------------------------------------------------                    
                
                ze_bel_cbh = np.zeros(len(ref_bel_cbh))
                valid_ref_id = np.where((~np.isnan(ref_bel_cbh)) & (ref_bel_cbh > -999.) )[0]
                
                # If no valid reflectivity, then the cloud is not precipitating
                if np.size(valid_ref_id) == 0.:
                    precip_binary[tt] = 0
                    T_mean_bel[tt] = np.nanmean(T_bel_cbh)
                # If there *is* valid reflectivity but the minimum
                # Doppler velocity is positive (updraft), then flag
                # this with precip_updraft_binary[tt] == 1
                elif (np.size(valid_ref_id) > 0.) and (np.nanmin(vel_bel_cbh[valid_ref_id]) >= 0.):
                    precip_binary[tt] = -555.
                    precip_updraft_binary[tt] = 1
                    T_mean_bel[tt] = np.nanmean(T_bel_cbh)
                # If there is both a valid reflectivity value *AND* the minimum
                # Doppler Velocity is negative
                elif (np.size(valid_ref_id) > 0.) and (np.nanmin(vel_bel_cbh[valid_ref_id]) < 0.):
                    # Convert from dBZ to linear Ze
                    ze_bel_cbh[valid_ref_id] = 10.**(ref_bel_cbh[valid_ref_id]/10.)
                    precip_updraft_binary[tt] = 0
                    tmp_T_mean = np.nanmean(T_bel_cbh)
                    # Mean Ze is calculate by summing only valid reflectivity bins
                    # and dividing by the the number of bins corresponding to dmin
                    tmp_ze_sum = np.nansum(ze_bel_cbh[valid_ref_id])
                    tmp_ze_mean = tmp_ze_sum/Radar_precip_depth_bin_thresh
                    # Convert back to dBZ
                    tmp_ref_mean = 10.*np.log10(tmp_ze_mean)
                    # Calculate mean Z_min
                    Z_min_ze = 10.**(Z_min_bel_cbh/10.)
                    Z_min_ze_mean = np.nanmean(Z_min_ze)
                    tmp_vel_min = np.nanmin(vel_bel_cbh[valid_ref_id])
                    Z_min_mean = 10.*np.log10(Z_min_ze_mean)
                    # If the layer is less than the minimum Ze threshold, we will
                    # not consider the cloud to be precipitating and will move to
                    # the next time step. Otherwise, precip_binary[tt] == 1 and
                    # we calculate the precipitation rates.
                    if not arb_dbz_cutoff:
                        if tmp_ref_mean < Z_min_mean:
                            precip_binary[tt] = 0.
                            #continue
                        else:
                            precip_binary[tt] = 1
                    elif arb_dbz_cutoff:
                        if tmp_ref_mean < -20.:
                            precip_binary[tt] = 0.
                            #continue
                        else:
                            precip_binary[tt] = 1     
                            
                    #-------------------------------------------------                    
                    #-------------------------------------------------                    
                    # Compute precipitation rates in mm/hr
                    #
                    # Check the temperature of the sub-cloud layer.
                    #
                    # If it is below 0 deg C, we will assume it to be
                    # subsaturated w.r.t. liquid and supersaturated
                    # w.r.t ice and use Hogan et al. (2006) [https://doi.org/10.1175/JAM2340.1]
                    # to compute temperature-dependent ice water content and multiply this by the
                    # minimum Doppler velocity within dmin to get the
                    # ice water flux. We use the 94 GHz parameterization from Hogan et al.
                    #
                    # If the subcloud layer is above 0 deg C, we will
                    # use the Comstock et al. (2004) [https://doi.org/10.1256/qj.03.187]
                    # drizzle Z-R relationship to compute the rain rate.
                    #-------------------------------------------------                    
                    #-------------------------------------------------                    
                    if False:
                        if precip_binary[tt] == 1:
                            # IWFlx (T < 0 deg C)
                            if tmp_T_mean < 0.:
                                tmp_IWflx = (10.**(5.8e-4*(10.*np.log10(10.**(tmp_ref_mean/10))) * tmp_T_mean + 9.23e-2 * (10.*np.log10(10.**(tmp_ref_mean/10))) - 7.06e-3 * tmp_T_mean - 0.992))
                                tmp_IWflx = tmp_IWflx*tmp_vel_min*(-3.6)
                                IWflx[tt] = tmp_IWflx

                            # LWflx (T >= 0 deg C)
                            elif tmp_T_mean >= 0.:
                                tmp_LWflx = ((10.**(tmp_ref_mean/10))/25.)**(1/1.3)
                                LWflx[tt] = (tmp_LWflx)

                    ref_mean_bel[tt] = tmp_ref_mean
                    vel_min_bel[tt] = tmp_vel_min
                    T_mean_bel[tt] = tmp_T_mean
                    Z_min_mean_bel[tt] = Z_min_mean
                    
                    #-------------------------------------------------                    
                    # Check to see if precipitation is continuous to the surface.
                    #
                    # If it is not and the Boolean calc_virga_props == True,
                    # then calculate the virga base height.
                    #
                    # Note that virga is calculated whether or not there
                    # is valid reflectivity at cloud base. This is to account
                    # for scenarios where a geometrically thin liquid cloud
                    # precipitates into a sub-cloud layer that is subsaturated
                    # w.r.t. liquid but supersaturated w.r.t ice and the ice
                    # grows and then sublimates. Probably a rare occurrence,
                    # but we still want to document it.
                    #-------------------------------------------------
                    if precip_binary[tt] == 1.:
                        # First, limit profiles to only those above hmin (162.5 m, midbin of 150-175 m range gates)
                        ref_below_cbh_prof_lim = ref_below_cbh_prof[min_basta_loc:]
                        height_below_cbh_prof_lim = height_below_cbh_prof[min_basta_loc:]
                        Z_min_below_cbh_prof_lim = Z_min_below_cbh_prof[min_basta_loc:]
                        # This line is only relevant for Z_min > -36 dBZ at 1 km.
                        ref_below_cbh_prof_lim[ref_below_cbh_prof_lim < Z_min_below_cbh_prof_lim] = -999.

                        # Boolean plot for checking these limited profiles relative to the CBH
                        if False:    
                            fig = plt.figure(figsize=(8,8))
                            ax=fig.add_subplot(111)
                            Fontsize=14
                            ax.grid(which='both')
                            ax.tick_params(labelsize=Fontsize)
                            ax.set_xlabel('Reflectivity [dBZ]',fontsize=Fontsize)
                            ax.set_ylabel('Height [m]',fontsize=Fontsize)
                            ax.plot(single_time_ref_prof,height,lw=3,c='navy',label='reflectivity')
                            ax.axhline(single_time_cbh,lw=3,c='darkorange',label='CBH')
                            ax.legend(loc='best',fontsize=Fontsize)
                            ax.set_ylim(0,5000)
                            ax.set_xlim(-50,10)
                            ax.set_title(time_dt[tt],fontsize=Fontsize*2.)
                            plt.show()
                            plt.close()


                        # Check to see if all values below CBH are valid reflectivity values
                        if np.all(ref_below_cbh_prof_lim > -999.) & np.all(~np.isnan(ref_below_cbh_prof_lim)):
                            precip_to_sfc_binary[tt] = 1
                            if calc_virga_props:
                                virga_binary[tt] = 0
                        else:

                            # Check for NaNs below CBH.
                            nanid = np.where(np.isnan(ref_below_cbh_prof_lim))
                            if np.size(nanid) > 0.:
                                # There are NaNs in the profile below CBH.
                                # Note that NaNs do not mean hydrometeor-free range gates but rather
                                # values that have been QC'd and determined to be "bad". In this case,
                                # we want at least 75% of the data below CBH to NOT NaNs in order
                                # to perform a computation.
                                frac_bad_size = np.size(nanid)/np.size(ref_below_cbh_prof_lim)
                                if frac_bad_size < 0.25:
                                    # Less than 25% of the data below cloud base are NaNs

                                    # First, grab a reflectivity profile that excludes the NaNs
                                    # If every value other than the NaNs are valid (i.e., not -999.)
                                    # then we will say the cloud is precipitating to the surface

                                    ref_below_cbh_prof_lim_no_nans = ref_below_cbh_prof_lim[~np.isnan(ref_below_cbh_prof_lim)]
                                    if np.all(ref_below_cbh_prof_lim_no_nans > -999.):
                                        precip_to_sfc_binary[tt] = 1
                                        if calc_virga_props:
                                            virga_binary[tt] = 0
                                    elif np.all(ref_below_cbh_prof_lim_no_nans == -999.):
                                        # Just a sanity check. This shouldn't happen.
                                        raise RuntimeError('Something went wrong')
                                    else:
                                        precip_to_sfc_binary[tt] = 0
                                        if calc_virga_props:
                                            virga_binary[tt] = 1
                                            # If here, then there are NaNs (but no more than 25%)
                                            # below cloud base and there are some valid hydrometeor-
                                            # containing range gates (> -999.). In thise case, we'll
                                            # make a cloud mask with NaNs, loop down from cloud base,
                                            # skip over the NaNs, and find the first cloud mask that is
                                            # 0. 
                                            ref_below_cbh_prof_lim_rev = np.flip(ref_below_cbh_prof_lim)
                                            height_below_cbh_prof_lim_rev = np.flip(height_below_cbh_prof_lim)
                                            # Note that these "virga" prefixes associate with reversed arrays
                                            virga_cloud_mask = np.zeros(np.shape(ref_below_cbh_prof_lim_rev))
                                            valid_virga_cloud_id = np.where(ref_below_cbh_prof_lim_rev > -999.)
                                            valid_virga_cloud_id = np.squeeze(valid_virga_cloud_id)
                                            virga_cloud_mask[valid_virga_cloud_id] = 1
                                            virga_nan_id = np.where(np.isnan(ref_below_cbh_prof_lim_rev))
                                            virga_cloud_mask[virga_nan_id] = np.nan

                                            # Limit virga cloud mask to only those values within depth dmin
                                            virga_cloud_mask_dmin = virga_cloud_mask[0:Radar_precip_depth_bin_thresh]
                                            virga_cloud_mask_id_dmin = np.where(virga_cloud_mask_dmin == 1.)
                                            virga_cloud_mask_id_dmin = np.squeeze(virga_cloud_mask_id_dmin)
                                            virga_cloud_mask_id_dmin_max = np.max(virga_cloud_mask_id_dmin) # value where last reflectivity point is valid within dmin depth below cloud base

                                            # Find all values in profile scanning from first valid reflectivity below cloud base *within dmin)
                                            # to the value where the virga_cloud mask == 0 minus 1 (kkk-1)
                                            for kkk in range(virga_cloud_mask_id_dmin_max,len(virga_cloud_mask)):
                                                if np.isnan(virga_cloud_mask[kkk]):
                                                    pass
                                                elif virga_cloud_mask[kkk] == 1.:
                                                    pass
                                                elif virga_cloud_mask[kkk] == 0.:
                                                    # limit virga cloud mask to only those values before reaching a "zero" (non-cloud)
                                                    virga_cloud_mask_lim = virga_cloud_mask[0:kkk]
                                                    virga_cloud_mask_height_lim = height_below_cbh_prof_lim_rev[0:kkk]
                                                    virga_cloud_mask_lim_id = np.where(virga_cloud_mask_lim == 1.)
                                                    virga_cloud_mask_lim_id = np.squeeze(virga_cloud_mask_lim_id)
                                                    last_valid_height_id = np.max(virga_cloud_mask_lim_id)
                                                    last_valid_height = virga_cloud_mask_height_lim[last_valid_height_id]
                                                    virga_base_height[tt] = last_valid_height   
                                                    break
                                else:
                                    # More than 75% of the data below cloud base are NaNs
                                    precip_to_sfc_binary[tt] = -555. # undetermined
                                    if calc_virga_props:
                                        virga_binary[tt] = -555. # undetermined                                
                            else:
                                # There are no NaNs in the profile below CBH, however
                                # not all bins have hydrometeors (i.e., some are -999.)
                                # In this case, precip does not reach the surface and we
                                # calculate virga base height

                                precip_to_sfc_binary[tt] = 0

                                if calc_virga_props:
                                    virga_binary[tt] = 1

                                    # reverse the array
                                    ref_below_cbh_prof_lim_rev = np.flip(ref_below_cbh_prof_lim)
                                    height_below_cbh_prof_lim_rev = np.flip(height_below_cbh_prof_lim)

                                    # There are cases where there is no reflectivity at cloud base
                                    # but there is precipitation below cloud base (perhaps
                                    # the formation of ice crystals below liquid cloud base?)
                                    virga_cloud_mask = np.zeros(np.shape(ref_below_cbh_prof_lim_rev))
                                    virga_cloud_mask_id = np.where(ref_below_cbh_prof_lim_rev > -999.)
                                    virga_cloud_mask_id = np.squeeze(virga_cloud_mask_id)
                                    virga_cloud_mask[virga_cloud_mask_id] = 1

                                    # Limit virga cloud mask to only those values within depth dmin
                                    virga_cloud_mask_dmin = virga_cloud_mask[0:Radar_precip_depth_bin_thresh]
                                    if np.size(virga_cloud_mask_dmin) > 1:
                                        virga_cloud_mask_id_dmin = np.where(virga_cloud_mask_dmin == 1.)
                                        virga_cloud_mask_id_dmin = np.squeeze(virga_cloud_mask_id_dmin)
                                        virga_cloud_mask_id_dmin_max = np.max(virga_cloud_mask_id_dmin) # value where last reflectivity point is valid within dmin depth below cloud base
                                    elif np.size(virga_cloud_mask_dmin) == 1.:
                                        virga_cloud_mask_id_dmin_max = 0
                                    # Find all values in profile scanning from first valid reflectivity below cloud base *within* dmin)
                                    # to the value where the virga_cloud mask == 0 minus 1 (kkk-1)
                                    for kkk in range(virga_cloud_mask_id_dmin_max,len(virga_cloud_mask)):
                                        if virga_cloud_mask[kkk] == 1.:
                                            pass
                                        elif virga_cloud_mask[kkk] == 0.:
                                            # limit virga cloud mask to only those values before reaching a "zero" (non-cloud)
                                            virga_cloud_mask_lim = virga_cloud_mask[0:kkk]
                                            virga_cloud_mask_height_lim = height_below_cbh_prof_lim_rev[0:kkk]
                                            virga_cloud_mask_lim_id = np.where(virga_cloud_mask_lim == 1.)
                                            virga_cloud_mask_lim_id = np.squeeze(virga_cloud_mask_lim_id)
                                            last_valid_height_id = np.max(virga_cloud_mask_lim_id)
                                            last_valid_height = virga_cloud_mask_height_lim[last_valid_height_id]
                                            virga_base_height[tt] = last_valid_height 
                                            break

                            

                #-------------------------------------------------
                #-------------------------------------------------
                #-------------------------------------------------
                # STEP 3: Calculate the presence and rates of
                # *surface* precipitation.
                #
                # This is done by looking at hmin + dmin *from the
                # surface*.
                # 
                # The only requirement for this step is that an 
                # overlying cloud layer exist (i.e., single_time_cbh 
                # >= min_cbh_thresh, which is the conditional block 
                # we are still in), whether or not that cloud
                # layer is precipitating. If it is not precipitating
                # from cloud base but there is precip at the surface,
                # then we want to know why, as this could potentially
                # be fog. Cases of fog will hopefully be filtered
                # using the above fog properties calculations.
                #-------------------------------------------------        
                #-------------------------------------------------        
                #-------------------------------------------------   
                
                # Limit profiles from hmin up to hmin+dmin
                tmp_sfc_ref = ref_below_cbh_prof[min_basta_loc:min_basta_loc+Radar_precip_depth_bin_thresh]
                tmp_sfc_vel = vel_below_cbh_prof[min_basta_loc:min_basta_loc+Radar_precip_depth_bin_thresh]
                tmp_sfc_T = T_below_cbh_prof[min_basta_loc:min_basta_loc+Radar_precip_depth_bin_thresh]
                tmp_sfc_height = height_below_cbh_prof[min_basta_loc:min_basta_loc+Radar_precip_depth_bin_thresh]
                tmp_sfc_Z_min = Z_min_below_cbh_prof[min_basta_loc:min_basta_loc+Radar_precip_depth_bin_thresh]
                #-------------------------------------------------                   
                # Compute surface precip occurrence
                #-------------------------------------------------   
                tmp_sfc_ze = np.zeros(len(tmp_sfc_ref))
                valid_ref_id = np.where((~np.isnan(tmp_sfc_ref)) & (tmp_sfc_ref > -999.) )[0]
                if np.size(valid_ref_id) == 0.:
                    sfc_precip_binary[tt] = 0
                    sfc_T_mean[tt] = np.nanmean(tmp_sfc_T)
                elif (np.size(valid_ref_id) > 0.) and (np.nanmin(tmp_sfc_vel[valid_ref_id]) >= 0.):
                    sfc_precip_binary[tt] = -555.
                    sfc_precip_updraft_binary[tt] = 1
                    sfc_T_mean[tt] = np.nanmean(tmp_sfc_T)                        
                elif (np.size(valid_ref_id) > 0.) and (np.nanmin(tmp_sfc_vel[valid_ref_id]) < 0.):
                    tmp_sfc_ze[valid_ref_id] = 10.**(tmp_sfc_ref[valid_ref_id]/10.)
                    sfc_precip_updraft_binary[tt] = 0
                    tmp_sfc_T_mean = np.nanmean(tmp_sfc_T)
                    tmp_sfc_ze_sum = np.nansum(tmp_sfc_ze[valid_ref_id])
                    tmp_sfc_ze_mean = tmp_sfc_ze_sum/Radar_precip_depth_bin_thresh
                    tmp_sfc_ref_mean = 10.*np.log10(tmp_sfc_ze_mean)
                    sfc_Z_min_ze = 10.**(tmp_sfc_Z_min/10.)
                    sfc_Z_min_ze_mean = np.nanmean(sfc_Z_min_ze)
                    tmp_sfc_vel_min = np.nanmin(tmp_sfc_vel[valid_ref_id])
                    tmp_sfc_Z_min_mean = 10.*np.log10(sfc_Z_min_ze_mean)
                    if not arb_dbz_cutoff:
                        if tmp_sfc_ref_mean < tmp_sfc_Z_min_mean:
                            sfc_precip_binary[tt] = 0.
                        else:
                            sfc_precip_binary[tt] = 1
                    elif arb_dbz_cutoff:
                        if tmp_sfc_ref_mean < -20.:
                            sfc_precip_binary[tt] = 0.
                        else:
                            sfc_precip_binary[tt] = 1        
                    #-------------------------------------------------                   
                    # Compute surface precip rates
                    #-------------------------------------------------  
                    if False:
                        # IWFlx (T < 0 deg C)
                        if tmp_sfc_T_mean < 0.:
                            tmp_sfc_IWflx = (10.**(5.8e-4*(10.*np.log10(10.**(tmp_sfc_ref_mean/10))) * tmp_sfc_T_mean + 9.23e-2 * (10.*np.log10(10.**(tmp_sfc_ref_mean/10))) - 7.06e-3 * tmp_sfc_T_mean - 0.992)) # IWC based on Hogan et al., 2006, for 94 GHz.
                            tmp_sfc_IWflx = tmp_sfc_IWflx*tmp_sfc_vel_min*(-3.6)
                            sfc_IWflx[tt] = tmp_sfc_IWflx

                        # LWflx (T >= 0 deg C)
                        elif tmp_sfc_T_mean >= 0.:
                            tmp_sfc_LWflx = ((10.**(tmp_sfc_ref_mean/10))/25.)**(1/1.3)
                            sfc_LWflx[tt] = (tmp_sfc_LWflx)

                    sfc_ref_mean[tt] = tmp_sfc_ref_mean
                    sfc_vel_min[tt] = tmp_sfc_vel_min
                    sfc_T_mean[tt] = tmp_sfc_T_mean
                    sfc_Z_min_mean[tt] = tmp_sfc_Z_min_mean
                else:
                    print(a)
                #-------------------------------------------------              
                #-------------------------------------------------                    
                #-------------------------------------------------                    
                # STEP A1: Calculate normalized profiles w.r.t. CBH
                # This requires that there be reflectivity
                # present at cloud base.
                #-------------------------------------------------    
                #-------------------------------------------------    
                #-------------------------------------------------    

                if calc_normalized_props and (ref_at_cbh_binary[tt] == 1) and (precip_binary[tt] == 1):
                    
                    # Reverse arrays for easier computation
                    ref_below_cbh_prof_rev = np.flip(ref_below_cbh_prof)
                    vel_below_cbh_prof_rev = np.flip(vel_below_cbh_prof)
                    T_below_cbh_prof_rev = np.flip(T_below_cbh_prof)
                    height_below_cbh_prof_rev = np.flip(height_below_cbh_prof)
                    
                    # Below CBH   
                    valid_ref_bel_id =  np.where( (~np.isnan(ref_below_cbh_prof_rev) & (ref_below_cbh_prof_rev > -999.) ) )
                    cloud_layer_mask_bel_rev = np.zeros(np.shape(ref_below_cbh_prof_rev))
                    cloud_layer_mask_bel_rev[valid_ref_bel_id] = 1
                    cloud_Objects_bel_rev,num_cloud_objects_bel = ndimage.label(cloud_layer_mask_bel_rev)
                    valid_ref_bel_first_layer_id = np.where(cloud_Objects_bel_rev == 1)[0]                

                    ref_norm_prof_bel[tt,valid_ref_bel_first_layer_id] = ref_below_cbh_prof_rev[valid_ref_bel_first_layer_id]
                    vel_norm_prof_bel[tt,valid_ref_bel_first_layer_id] = vel_below_cbh_prof_rev[valid_ref_bel_first_layer_id]
                    T_norm_prof_bel[tt,valid_ref_bel_first_layer_id] = T_below_cbh_prof_rev[valid_ref_bel_first_layer_id]
                    height_norm_prof_bel[tt,valid_ref_bel_first_layer_id] = height_below_cbh_prof_rev[valid_ref_bel_first_layer_id]                                     
            
                    # Above CBH
                    valid_ref_above_id =  np.where( (~np.isnan(ref_above_cbh_prof) & (ref_above_cbh_prof > -999.) ) )
                    cloud_layer_mask_above = np.zeros(np.shape(ref_above_cbh_prof))
                    cloud_layer_mask_above[valid_ref_above_id] = 1
                    cloud_Objects_above,num_cloud_objects_above = ndimage.label(cloud_layer_mask_above)
                    valid_ref_above_first_layer_id = np.where(cloud_Objects_above == 1)[0]                                               
                                          
                    ref_norm_prof_above[tt,valid_ref_above_first_layer_id] = ref_above_cbh_prof[valid_ref_above_first_layer_id]
                    vel_norm_prof_above[tt,valid_ref_above_first_layer_id] = vel_above_cbh_prof[valid_ref_above_first_layer_id]
                    T_norm_prof_above[tt,valid_ref_above_first_layer_id] = T_above_cbh_prof[valid_ref_above_first_layer_id]
                    height_norm_prof_above[tt,valid_ref_above_first_layer_id] = height_above_cbh_prof[valid_ref_above_first_layer_id]
                                  
                #-------------------------------------------------              
                #-------------------------------------------------                    
                #-------------------------------------------------                    
                # End Cases with single_time_cbh > min_thresh
                #-------------------------------------------------    
                #-------------------------------------------------    
                #-------------------------------------------------                         

            #-------------------------------------------------              
            #-------------------------------------------------                    
            #-------------------------------------------------                    
            # End Cases with ~np.isnan(single_time_cbh)
            #-------------------------------------------------    
            #-------------------------------------------------    
            #-------------------------------------------------  

        #-------------------------------------------------              
        #-------------------------------------------------                    
        #-------------------------------------------------                    
        # End tt loop
        #-------------------------------------------------    
        #-------------------------------------------------    
        #-------------------------------------------------             

    
    #-------------------------------------------------              
    #-------------------------------------------------                    
    #-------------------------------------------------                    
    # STEP 4: Begin array calculations for properties
    # calculated through the 'tt' time loop
    #-------------------------------------------------    
    #-------------------------------------------------    
    #-------------------------------------------------       

                                            
    #-------------------------------------------------       
    # Combine above and below cbh profiles such that 
    # they are normalized by cbh
    #-------------------------------------------------       
                                            
    if calc_normalized_props:
        height_above_cbh = np.arange(0,500*25.,25.)
        height_below_cbh = np.arange(-25,-501*25.,-25.)
        height_norm_cbh = np.concatenate((np.flip(height_below_cbh),height_above_cbh))

        ref_norm_cbh = np.zeros((num_times,1000))-999.
        vel_norm_cbh = np.zeros((num_times,1000))-999.
        T_norm_cbh = np.zeros((num_times,1000))-999.

        for tt in range(num_times):
            tmp_ref_norm_prof_above = ref_norm_prof_above[tt,:]
            tmp_vel_norm_prof_above = vel_norm_prof_above[tt,:]
            tmp_T_norm_prof_above = T_norm_prof_above[tt,:]
            tmp_ref_norm_prof_bel = ref_norm_prof_bel[tt,:]
            tmp_vel_norm_prof_bel = vel_norm_prof_bel[tt,:]
            tmp_T_norm_prof_bel = T_norm_prof_bel[tt,:]

            tmp_ref_norm_cbh = np.concatenate((np.flip(tmp_ref_norm_prof_bel),tmp_ref_norm_prof_above))
            tmp_vel_norm_cbh = np.concatenate((np.flip(tmp_vel_norm_prof_bel),tmp_vel_norm_prof_above))
            tmp_T_norm_cbh = np.concatenate((np.flip(tmp_T_norm_prof_bel),tmp_T_norm_prof_above))

            ref_norm_cbh[tt,:] = tmp_ref_norm_cbh
            vel_norm_cbh[tt,:] = tmp_vel_norm_cbh
            T_norm_cbh[tt,:] = tmp_T_norm_cbh
        
    #-------------------------------------------------       
    # Add all precip rates to rcb
    #-------------------------------------------------       
    if False:
        # Cloud base
        rcb = np.zeros(num_times)-999.
        ice_id = np.where(IWflx > -999.)[0]
        rcb[ice_id] = IWflx[ice_id]
        liq_id = np.where(LWflx > -999.)[0]
        rcb[liq_id] = LWflx[liq_id]  

        # Surface
        sfc_rcb = np.zeros(num_times)-999.
        sfc_ice_id = np.where(sfc_IWflx > -999.)[0]
        sfc_rcb[sfc_ice_id] = sfc_IWflx[sfc_ice_id]
        sfc_liq_id = np.where(sfc_LWflx > -999.)[0]
        sfc_rcb[sfc_liq_id] = sfc_LWflx[sfc_liq_id]                                              
                                            
    #-------------------------------------------------
    # Boolean plot to check precip rate time series
    #-------------------------------------------------           
    if False:
        dfmt = mdates.DateFormatter('%d-%H')
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(111)
        dum_LWflx = LWflx.copy()
        dum_LWflx[dum_LWflx == -999.] = np.nan
        ax1.plot(time_dt,dum_LWflx,color='red',lw=2)
        dum_IWflx = IWflx.copy()
        dum_IWflx[dum_IWflx == -999.] = np.nan
        ax1.plot(time_dt,dum_IWflx,color='blue',lw=2)                
        ax1.set_yscale('log')
        ax1.xaxis.set_major_formatter(dfmt) 
        ax1.set_ylabel('R$_{cb}$ [mm hr$^{-1}$]')
        ax1.set_xlabel('Time UTC [day-hour]')
        plt.show()
        plt.close()    
        
        
    #-------------------------------------------------       
    # Misc.
    #-------------------------------------------------         
    three_cbh_id = np.where( (~np.isnan(ceil_cbh_1)) & (~np.isnan(ceil_cbh_2)) & (~np.isnan(ceil_cbh_3)) )
    two_cbh_id = np.where( (~np.isnan(ceil_cbh_1)) & (~np.isnan(ceil_cbh_2)) & (np.isnan(ceil_cbh_3)) )
    one_cbh_id = np.where( (~np.isnan(ceil_cbh_1)) & (np.isnan(ceil_cbh_2)) & (np.isnan(ceil_cbh_3)) )
    num_ceil_cbh_in_profile[three_cbh_id] = 3
    num_ceil_cbh_in_profile[two_cbh_id] = 2
    num_ceil_cbh_in_profile[one_cbh_id] = 1
    

    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    # STEP 5: Write retrievals to dictionary
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    props_dict[dmin_str][ze_thresh_str]['time'] = time_dt
        
    #-------------------------------------------------        
    # Precip rates
    #-------------------------------------------------
    #props_dict[dmin_str][ze_thresh_str]['IWflx'] = IWflx
    #props_dict[dmin_str][ze_thresh_str]['sfc_IWflx'] = sfc_IWflx
    #props_dict[dmin_str][ze_thresh_str]['LWflx'] = LWflx
    #props_dict[dmin_str][ze_thresh_str]['sfc_LWflx'] = sfc_LWflx
    #props_dict[dmin_str][ze_thresh_str]['rcb'] = rcb
    #props_dict[dmin_str][ze_thresh_str]['sfc_rcb'] = sfc_rcb

    #-------------------------------------------------        
    # Cloud macrophysics
    #-------------------------------------------------    
    props_dict[dmin_str][ze_thresh_str]['ctt'] = ctt
    props_dict[dmin_str][ze_thresh_str]['cbt'] = cbt
    props_dict[dmin_str][ze_thresh_str]['cbh'] = cbh
    props_dict[dmin_str][ze_thresh_str]['cth'] = cth
    props_dict[dmin_str][ze_thresh_str]['c_thick'] = c_thick
    
    #-------------------------------------------------        
    # Sub-cloud layer and surface layer properties
    #-------------------------------------------------
    props_dict[dmin_str][ze_thresh_str]['ref_mean_bel'] = ref_mean_bel
    props_dict[dmin_str][ze_thresh_str]['vel_min_bel'] = vel_min_bel
    props_dict[dmin_str][ze_thresh_str]['T_mean_bel'] = T_mean_bel
    props_dict[dmin_str][ze_thresh_str]['Z_min_mean_bel'] = Z_min_mean_bel
    
    props_dict[dmin_str][ze_thresh_str]['sfc_ref_mean'] = sfc_ref_mean
    props_dict[dmin_str][ze_thresh_str]['sfc_vel_min'] = sfc_vel_min
    props_dict[dmin_str][ze_thresh_str]['sfc_T_mean'] = sfc_T_mean
    props_dict[dmin_str][ze_thresh_str]['sfc_Z_min_mean'] = sfc_Z_min_mean    
    
    #-------------------------------------------------        
    # Binaries
    #-------------------------------------------------    
    props_dict[dmin_str][ze_thresh_str]['cloud_binary'] =  cloud_binary
    props_dict[dmin_str][ze_thresh_str]['precip_binary'] =  precip_binary
    props_dict[dmin_str][ze_thresh_str]['sfc_precip_binary'] =  sfc_precip_binary
    props_dict[dmin_str][ze_thresh_str]['precip_to_sfc_binary'] =  precip_to_sfc_binary
    props_dict[dmin_str][ze_thresh_str]['precip_updraft_binary'] =  precip_updraft_binary
    props_dict[dmin_str][ze_thresh_str]['sfc_precip_updraft_binary'] =  sfc_precip_updraft_binary
    props_dict[dmin_str][ze_thresh_str]['single_layer_binary'] =  single_layer_binary
    props_dict[dmin_str][ze_thresh_str]['cbh_bel_min_binary'] =  cbh_bel_min_binary
    props_dict[dmin_str][ze_thresh_str]['ref_at_cbh_binary'] =  ref_at_cbh_binary
    props_dict[dmin_str][ze_thresh_str]['ref_within_100m_above_cbh_binary'] =  ref_within_100m_above_cbh_binary
    props_dict[dmin_str][ze_thresh_str]['ref_within_100m_below_cbh_binary'] =  ref_within_100m_below_cbh_binary
    props_dict[dmin_str][ze_thresh_str]['ref_above_cbh_binary'] =  ref_above_cbh_binary
    props_dict[dmin_str][ze_thresh_str]['ref_below_cbh_binary'] =  ref_below_cbh_binary 
    props_dict[dmin_str][ze_thresh_str]['cloud_50m_thick_binary'] =  cloud_50m_thick_binary
    
    #-------------------------------------------------        
    # Misc. Cloud Properties
    #-------------------------------------------------        
    props_dict[dmin_str][ze_thresh_str]['num_ref_cloud_layers_in_profile'] = num_ref_cloud_layers_in_profile
    props_dict[dmin_str][ze_thresh_str]['num_ceil_cbh_in_profile'] = num_ceil_cbh_in_profile
    props_dict[dmin_str][ze_thresh_str]['nearest_ref_above_cbh'] = nearest_ref_above_cbh
    props_dict[dmin_str][ze_thresh_str]['nearest_ref_below_cbh'] = nearest_ref_below_cbh

    #-------------------------------------------------        
    # Surface Met.
    #-------------------------------------------------
    if sfc_present_flag > 0.:
        props_dict[dmin_str][ze_thresh_str]['sfc_rh'] = sfc_rh
        #props_dict[dmin_str][ze_thresh_str]['sfc_temperature'] = sfc_temperature
        #props_dict[dmin_str][ze_thresh_str]['sfc_pressure'] = sfc_pressure
        #props_dict[dmin_str][ze_thresh_str]['sfc_wind_speed'] = sfc_wind_speed
        #props_dict[dmin_str][ze_thresh_str]['sfc_wind_dir'] = sfc_wind_dir
    
    #-------------------------------------------------        
    # Normalized properties
    #-------------------------------------------------     
    if calc_normalized_props:
        props_dict[dmin_str][ze_thresh_str]['ref_norm_prof_above'] = ref_norm_prof_above
        props_dict[dmin_str][ze_thresh_str]['vel_norm_prof_above'] = vel_norm_prof_above
        props_dict[dmin_str][ze_thresh_str]['T_norm_prof_above'] = T_norm_prof_above
        props_dict[dmin_str][ze_thresh_str]['height_norm_prof_above'] = height_norm_prof_above
        props_dict[dmin_str][ze_thresh_str]['ref_norm_prof_bel'] = ref_norm_prof_bel
        props_dict[dmin_str][ze_thresh_str]['vel_norm_prof_bel'] = vel_norm_prof_bel
        props_dict[dmin_str][ze_thresh_str]['T_norm_prof_bel'] = T_norm_prof_bel
        props_dict[dmin_str][ze_thresh_str]['height_norm_prof_bel'] = height_norm_prof_bel
        props_dict[dmin_str][ze_thresh_str]['ref_norm_cbh'] = ref_norm_cbh
        props_dict[dmin_str][ze_thresh_str]['vel_norm_cbh'] = vel_norm_cbh
        props_dict[dmin_str][ze_thresh_str]['T_norm_cbh'] = T_norm_cbh
        props_dict[dmin_str][ze_thresh_str]['height_norm_cbh'] = height_norm_cbh
    
    #-------------------------------------------------        
    # Virga
    #-------------------------------------------------     
    if calc_virga_props:
        props_dict[dmin_str][ze_thresh_str]['virga_base_height'] = virga_base_height
        props_dict[dmin_str][ze_thresh_str]['virga_binary'] = virga_binary

    #-------------------------------------------------        
    # Fog
    #-------------------------------------------------     
    if calc_fog_props:              
        props_dict[dmin_str][ze_thresh_str]['fog_binary'] = fog_binary
        props_dict[dmin_str][ze_thresh_str]['fog_possible_binary'] = fog_possible_binary

    #-------------------------------------------------        
    # Misc. Properties
    #-------------------------------------------------
    if sfc_present_flag == 1.:
        sfc_present_bool = True
    elif sfc_present_flag == 0.:
        sfc_present_bool = False
        
    #props_dict[dmin_str][ze_thresh_str]['cluster_id'] = cluster_id                                    
    #props_dict[dmin_str][ze_thresh_str]['native_cbh_1'] = ceil_cbh_1                                              
    #props_dict[dmin_str][ze_thresh_str]['native_cbh_2'] = ceil_cbh_2                                              
    #props_dict[dmin_str][ze_thresh_str]['native_cbh_3'] = ceil_cbh_3                                              
    props_dict[dmin_str][ze_thresh_str]['ceil_detection_status'] = ceil_detection_status                                            
    #props_dict[dmin_str][ze_thresh_str]['sounding_temperature'] = sounding_temperature                                          
    props_dict[dmin_str][ze_thresh_str]['sonde_sufficient_mask'] = sonde_sufficient_mask                                          
    props_dict[dmin_str][ze_thresh_str]['bad_radar_data_mask'] = bad_radar_data_flag                                        
    props_dict[dmin_str][ze_thresh_str]['flags'] = {'sfc_present_flag':sfc_present_bool,\
                                                    'calc_normalized_props':calc_normalized_props,\
                                                    'calc_fog_props':calc_fog_props,\
                                                    'calc_virga_props':calc_virga_props,\
                                                    'arb_dbz_cutoff':arb_dbz_cutoff,\
                                                    }
    props_dict[dmin_str][ze_thresh_str]['parameters'] = {'dmin':dmin,\
                                                    'ze_thresh':ze_thresh,\
                                                    'min_basta_loc':min_basta_loc,\
                                                    'hmin':hmin,\
                                                    'Radar_bin_width':Radar_bin_width,\
                                                    'min_cbh_thresh':min_cbh_thresh,\
                                                    'min_sep_for_cat':min_sep_for_cat,\
                                                    'Z_min':Z_min,\
                                                    }    
                                     
    date_out_str = date.strftime("%m_%d_%Y")
    #save_path = '/mnt/raid/mwstanfo/micre_cloud_properties/'
    #pkl_file = save_path+'micre_cloud_properties_dmin{}_ze_thresh{}_{}.p'.format(dmin_str,ze_thresh_str,date_out_str)
    #pickle.dump(props_dict,open(pkl_file,"wb"))
    return props_dict


