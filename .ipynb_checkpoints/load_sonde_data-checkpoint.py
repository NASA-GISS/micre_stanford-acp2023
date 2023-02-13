#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# load_sonde_data.py
# Original creator: Israel Silber
# Converted from Matlab to Python by McKenna Stanford
# Obtained by MS on 09-02-2020
# Last update: 10-20-2020
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#======================================================
# Python Imports
import numpy as np
from file_struct import file_struct as fs
from give_me_files_and_subfolders import give_me_files_and_subfolders
import xarray
import datetime
#======================================================
    
def load_sonde_data(flist_struct):
    
    """
    This function loads sounding data from files produced by the
    Australian Antarctic Division (AAD), specificaly for 
    Macquarie Island soundings during the MICRE field campaign.
    
    The function will load the data and save it into a dictionary
    called "Sonde" which will have the following keys (followed
    by a hash and a short description, including units):
    dewpoint_temp # dewpoint temperature in K
    drybulb_temp # drybulb temperature in K
    RH # relative humidity in %
    pressure # pressure in hPa
    wind_direction # wind direction in degrees
    wind_speed # wind speed in m/s
    u_wind # zonal component of wind in m/s
    v_wind # meridional component of wind in m/s
    lat # latitude in degrees
    lon # longitude in degrees
    alt # altitude in km
    time # elapsed time in a list of datetime objects
    ascent_rate # acent rate in m/s
    site_lat # site latitude in degrees
    site_lon # site longitude in degrees
    sfc_pressure # surface pressure in hPa
    sfc_temp # surface temperature in K
    sfc_humidity # surface humidity in %
    sfc_wind_speed # surface wind speed in m/s
    sfc_wind_direction # surface wind direction in degrees
    
    The dictionary key 'units' and 'long_name' can be accessed
    to retrieve this information.
    
    """
    
    siteName = 'macquarieisland'
    filePath = flist_struct.fpath
    fileName = flist_struct.fname

    
    # Create empty dictionary in which the 
    # final variables will be stored.
    Sonde = {}
    
    # Open NetCDF file
    ncfile = xarray.open_dataset(filePath+fileName)
    
    # Get attributes
    attrs = ncfile.attrs

    #----------------------------------
    # Get some attribute variables
    #----------------------------------
    site_lat = attrs['Site_Latitude']
    site_lon = attrs['Site_Longitude']
    sfc_pressure = attrs['Surface_Pressure']
    sfc_temp = attrs['Surface_Temperature']
    sfc_humidity = attrs['Surface_Humidity']
    sfc_wind_speed = attrs['Surface_Wind_Speed']
    sfc_wind_direction = attrs['Surface_Wind_Direction']
    
    #----------------------------------
    # Surface measurements
    #----------------------------------
    Sonde['site_lat'] = site_lat # degrees
    Sonde['site_lon'] = site_lon # degrees
    Sonde['sfc_pressure'] = sfc_pressure # hPa
    Sonde['sfc_temp'] = sfc_temp # K
    Sonde['sfc_humidity'] = sfc_humidity # RH
    Sonde['sfc_wind_speed'] = sfc_wind_speed # m/s
    Sonde['sfc_wind_direction'] = sfc_wind_direction # degrees
    
    #----------------------------------
    # Sonde measurements.
    #----------------------------------
    Sonde['dewpoint_temp'] = np.array(ncfile['DewPoint']) # in K
    Sonde['drybulb_temp'] = np.array(ncfile['Temperature']) # in K
    Sonde['RH'] = np.array(ncfile['Rel_humidity']) # in %
    Sonde['pressure'] = np.array(ncfile['Pressure']) # in hPa 
    Sonde['wind_direction'] = np.array(ncfile['Wind_Dir']) # in degrees
    Sonde['u_wind'] = np.array(ncfile['V_Zonal']) # in m/s
    Sonde['v_wind'] = np.array(ncfile['V_Merid']) # in m/s
    Sonde['wind_speed'] = np.array(ncfile['Wind_Speed']) # in m/s
    Sonde['ascent_rate'] = np.array(ncfile['Ascent_Rate']) # in m/s

    #----------------------------------   
    # radiosonde location.
    #----------------------------------
    Sonde['lat'] = np.array(ncfile['Lats']) # in degrees
    Sonde['lon'] = np.array(ncfile['Lons']) # in degrees
    Sonde['alt'] = np.array(ncfile['Altitude']) # in km
    

    #----------------------------------   
    # 'nan'ing bad data.
    #---------------------------------
    Sonde['dewpoint_temp'][Sonde['dewpoint_temp'] <= -999.99] = np.nan
    Sonde['drybulb_temp'][Sonde['drybulb_temp'] <= -999.99] = np.nan
    Sonde['RH'][Sonde['RH'] <= -999.99] = np.nan
    Sonde['pressure'][Sonde['pressure'] <= -999.99] = np.nan
    Sonde['wind_direction'][Sonde['wind_direction'] <= -999.99] = np.nan
    Sonde['wind_speed'][Sonde['wind_speed'] <= -999.99] = np.nan
    Sonde['v_wind'][Sonde['v_wind'] <= -999.99] = np.nan
    Sonde['u_wind'][Sonde['u_wind'] <= -999.99] = np.nan
    Sonde['ascent_rate'][Sonde['ascent_rate'] <= -999.99] = np.nan
    
    #-------------------------------------------    
    # From attributes, get initial time
    #-------------------------------------------
    tmp_init_time = attrs['Time']
    
    # init time is a string, so separate first by spaces--
    # the first index will be the time in HH:MM:SS, the
    # second index will be the date in DD/MM/YYYY, and the
    # third index should always be 'UT'
    time_split = str.split(tmp_init_time,' ')
    time_hhmmss = time_split[0]
    time_hhmmss_split = str.split(time_hhmmss,':')
    time_date = time_split[1]
    time_date_split = str.split(time_date,'/')
    
    init_time = datetime.datetime(int(time_date_split[2]),int(time_date_split[1]),\
                                  int(time_date_split[0]),int(time_hhmmss_split[0]),\
                                  int(time_hhmmss_split[1]),int(time_hhmmss_split[2]))
    

    #-------------------------------------------
    # Converting the 'elapsed_time' variable
    # into list of datetime objects
    #-------------------------------------------
    tmp_elapsed_time = np.array(ncfile['Elapsed_Time']) # in s
    
    time_deltas = [datetime.timedelta(seconds=int(tmp_elapsed_time[ii])) for ii in range(len(tmp_elapsed_time))]
    elapsed_time = [init_time + d for d in time_deltas]
    
    # write to dictionary
    Sonde['time'] = elapsed_time 
    
    #-------------------------------------------
    # Add dictionary key establishing units and
    # long names, title "units" and "long_name",
    # respectively.
    #-------------------------------------------
    Sonde['units'] = {'dewpoint_temp':'K',\
                      'drybulb_temp':'K',\
                      'RH':'%',\
                      'pressure':'hPa',\
                      'wind_direction':'degrees',\
                      'wind_speed':'m/s',\
                      'u_wind':'m/s',\
                      'v_wind':'m/s',\
                      'ascent_rate':'m/s',\
                      'latitude':'degrees',\
                      'longitude':'degrees',\
                      'altitude':'km',\
                      'time':'datetime object',\
                      'site_lat':'degrees',\
                      'site_lon':'degrees',\
                      'sfc_pressure':'hPa',\
                      'sfc_temp':'K',\
                      'sfc_humidity':'%',\
                      'sfc_wind_speed':'m/s',\
                      'sfc_wind_direction':'degrees',\
                     }
    
    Sonde['long_name'] = {'dewpoint_temp':'dew point temperature',\
                      'drybulb_temp':'dry bulb temperature',\
                      'RH':'relative humidity',\
                      'pressure':'pressure',\
                      'wind_direction':'wind direction',\
                      'wind_speed':'wind speed',\
                      'u_wind':'zonal component of wind',\
                      'v_wind':'meridional component of wind',\
                      'ascent_rate':'sonde ascent rate',\
                      'latitude':'sonde latitude',\
                      'longitude':'sonde longitude',\
                      'altitude':'sonde altitude',\
                      'time':'elapsed time since sonde release',\
                      'site_lat':'latitude of site',\
                      'site_lon':'longitude of site',\
                      'sfc_pressure':'surface pressure',\
                      'sfc_temp':'surface temperature',\
                      'sfc_humidity':'surface relative humidity',\
                      'sfc_wind_speed':'surface wind speed',\
                      'sfc_wind_direction':'surface wind direction',\
                     }
        
    # temporary
    file_keys = ncfile.keys()
    return Sonde
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TESTING
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# if False:
#     parent_path = '/Volumes/STANFORD_1/micre_soundings/'
#     search_string = '201605'
#     flist = give_me_files_and_subfolders(search_string,parent_path)
#     #flist = sorted(flist)
#     for ii in range(len(flist)):
#         print(flist[ii].fname)
           
# for kk in range(len(flist)):
#     tmp = load_sonde_data(flist[kk])
#     print(flist[kk].fname)
#     #print(np.min(tmp['alt']))
#     #print(np.max(tmp['alt']))
#     #print(tmp['alt'][0:5])
#     for key,val in tmp.items():
#         print(key,'max: '+str(np.max(val)),'min: '+str(np.min(val)))
#     print(aaaaa)
#     print(' ')
#     if kk == 3:
#         print(aaaaa)

