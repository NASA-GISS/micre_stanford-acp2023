#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate_theta_and_more.py 
# Original creator: Israel Silber
# Converted from Matlab to Python by McKenna Stanford
# Obtained by MS on 09-02-2020
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# {Output_dict} = calculate_theta_and_more(T, p, **kwargs)
#
# The function receives at least 3 parameters and calculates others.
# Function developed for the use of sounding data
#
# Input:
# T - temperatures in Celsius (can be an array); REQUIRED
# p - pressure in mbar (can be an array); REQUIRED
# RH - relative humidity in % (can be an array)
#    - Either RH or w is required--SPECIFY!
#    - Supply as keyword argument "RH"
# water_w - water vapor mixing ratio [g/kg] (can be an array)
#    - Either RH or w is required--SPECIFY!
#    - Supply as keyword argument "w"
# sat_pres_formula - saturation pressure formula keyword argument.
#    - supply one of the following keyword argument-value pairs.
#    - default is Flatau
#    -- sat_pres_formula='Flatau'
#    -- sat_pres_formula='Teten'
#    -- sat_pres_formula='Alduchov_Eskridge'
#    -- sat_pres_formula='Murphy_Koop'
#    -- sat_pres_formula='Emmanuel'
# use_T_K - keyword argument used to pass temperature in K;
#    - default is to supply T in Celsius
#    - set to True to supply in K
# use_pres_Pa - keyword argument used to pass pressure in Pa;
#    - default is to supply pressure in hPa (mb)
#    - set to True to supply in Pa
# use_w_kg_per_kg - keyword argument used to pass w in kg/kg;
#    - default is to supply pressure in g/kg
#    - set to True to supply in kg/kg'
#
# theta_e_un - Theta_e uncertainties
#    If the keyword argument "theta_e_un" is supplied, uncertainties
#    in Theta_e are calculated, but one must supply a tuple or 2D
#    array in order to due so. If a tuple, the 0th index is delta_Tk
#    and the 1st index is delta_RH. If a 2D array, the 0th dimension
#    is delta_Tk and the 1st dimesion is delta_RH.
#    If this keyword argument is not supplied, then Theta_e uncertainty 
#    is not calculated.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output dictionary contains:
# L_v - latent heat of vaporization (as it is temperature dependent).
# L_s - latent heat of sublimation (as it is temperature dependent).
# e_s - water vapor saturation pressure above liquid water.
# e_i - water vapor saturation pressure above ice water.
# e - water vapor pressure.
# w - water vapor mixing ratio.
# Theta - potential temperature.
# Theta_e - equivalent potential temperature.
# Theta_v - virtual potential temperature.
# RH_l - relative humidity with respect to liquid water
# RH_i - relative humidity with respect to ice
# w_s - saturation water vapor pressure
# q - specific humidity
# T_v - virtual temperature
# var_full_names - dictionary that contains the full names of each key (variable)
# var_units - dictionary that contains the units of each key (variable)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#======================================================
# Python Imports
import numpy as np
#======================================================

# NOTE: Using **kwargs requires that keywords and associated values be specified. Here, you
# ***must*** specify either "RH" or "w" as a keyword argument.
# Use "calculate_theta_and_more.__doc__" to print the docstring associated with this function

def calculate_theta_and_more(T,p,**kwargs):
    
    """
    This function calculates numerous thermodynamic parameters
    by supplying temperature, pressure, and either relative humidity
    or water vapor mixing ratio as a measure of humidity.
    Supply either RH or w is REQUIRED.
    
    This program assumes that temperature is being passed in Celsius,
    that pressure is being passed in hPa (mb), and that either RH is
    being passed in % or w (water vapor mixing ratio) is being passed
    in g/kg. 
    
    If the user wishes to supply temperature in K, pressure in Pa,
    or w in kg/kg, they may do so by supplying the following keyword
    arguments and setting them to True.
    
        use_T_K=True --> supply temperature in Kelvin
        use_w_kg_per_kg=True --> supply w in kg/kg
        use_pres_Pa=True --> supply pressure in Pa
    
    RH must always be supplied as a percentage.
    
    Input can be integers, lists, or numpy.ndarrays.
    
    Output will be supplied as integers or numpy.ndarrays.
    
    User can supply the preferred calculation for saturation vapor pressure
    using the 'sat_pres_formula' keyword argument and one of the following
    5 options:
        sat_pres_formula='Flatau'
        sat_pres_formula='Teten'
        sat_pres_formula='Alduchov_Eskridge'
        sat_pres_formula='Murphy_Koop'
        sat_pres_formula='Emmanuel'
    The default option is that of Flatau et al. (1992)

    If the keyword argument "theta_e_un" is supplied, uncertainties
    in Theta_e are calculated, but one must supply a tuple or 2D
    array in order to due so. If a tuple, the 0th index is delta_Tk
    and the 1st index is delta_RH. If a 2D array, the 0th dimension
    is delta_Tk and the 1st dimesion is delta_RH.
    If this keyword argument is not supplied, then Theta_e uncertainty 
    is not calculated.
    
    Output is in the form of a dictionary with the following parameters:
        L_v - latent heat of vaporization (as it is temperature dependent).
        L_s - latent heat of sublimation (as it is temperature dependent).
        e_s - water vapor saturation pressure above liquid water.
        e_i - water vapor saturation pressure above ice water.
        e - water vapor pressure.
        w - water vapor mixing ratio.
        Theta - potential temperature.
        Theta_e - equivalent potential temperature.
        Theta_v - virtual potential temperature.
        RH_l - relative humidity with respect to liquid water
        RH_i - relative humidity with respect to ice
        w_s - saturation water vapor pressure
        q - specific humidity
        T_v - virtual temperature
        var_full_names - dictionary that contains the full names of each key (variable)
        var_units - dictionary that contains the units of each key (variable)
    
    
    """
    
    varargin = kwargs
    
    #--------------------------------------------------------------------
    # Check to see if RH or water vapor mixing ratio is being supplied.
    # Kill program if neither is supplied.
    #--------------------------------------------------------------------    
    #result = True if "w" in varargin.keys() else False
    #if result == True:
    #    water_w = varargin['w']
    #else:
    #    RH = varargin['RH']
    if "w" in varargin.keys():
        water_w = varargin['w']
    elif "RH" in varargin.keys():
        RH = varargin['RH']
    else:
        raise RuntimeError('Did not supply either RH or vapor mixing ratio, please retry')
    
        
    #--------------------------------------------------------------------    
    # Check to see if theta_e uncertainties have been provided
    #--------------------------------------------------------------------
    
    result = True if "theta_e_un" in varargin.keys() else False
    if result == True:
        # assuming that the delta_var arguments will be help
        # as a tuple in which we can index as follows:
        delta_Tk = varargin['theta_e_un'][0]
        delta_RH = varargin['theta_e_un'][1]
    else:
        delta_Tk = None
        delta_RH = None
        

    #--------------------------------------------------------------------
    # Check to see if specific saturation pressure calculation is desired, 
    # and if so, choose which calculation to use. Default is Flatau et al. (1992)
    #    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
    #    Flatau: Flatau et al. (1992) coefficients for satruation pressure
    #    Alduchov_Eskridge: Use Alduchov and Eskridge (1996) empirical equations.
    #    Murphy_Koop:  Use Murphy and Koop (2005) empirical equations
    #    Teten: Use Teten's formula (ERA-Interim).
    #    Emmanuel: Emanuel (1994) (AMPS)
    #--------------------------------------------------------------------
    result = True if "sat_pres_formula" in varargin.keys() else False
    if result == True:
        sat_pres_formula = varargin['sat_pres_formula']
    else:
        sat_pres_formula = None
        
        
    #--------------------------------------------------------------------    
    # Check to see input argument type of T, raise error if not provided
    # as an integer, list, or numpy.ndarray. Use this information to
    # then convert T, p and RH/w to numpy.ndarray if supplied as a list.
    # 
    # Also use this opportunity to convert w to kg/kg if g/kg is supplied
    # (default). If keyword "use_w_kg_per_kg" is supplied, don't convert.
    #--------------------------------------------------------------------
    arg_type = type(T)
    
    if arg_type is int:
        #print('arguments supplied as integers')
        if 'w' in varargin.keys():
            if 'use_w_kg_per_kg' not in varargin.keys():
                water_w = water_w/1000.
            
    elif arg_type is list:
        #print('arguments supplied as lists, converting to numpy.ndarray')
        T = np.array(T)
        p = np.array(p)
        if 'RH' in varargin.keys():
            RH = np.array(RH)
        elif 'w' in varargin.keys():
            if 'use_w_kg_per_kg' not in varargin.keys():
                water_w = np.array(water_w)/1000.
            else:
                water_w = np.array(water_w)
            
    elif arg_type is np.ndarray:
        #print('arguments supplied as a numpy.ndarray')
        if 'w' in varargin.keys():
            if 'use_w_kg_per_kg' not in varargin.keys():
                water_w = water_w/1000.
            
    else:
        raise RuntimeError('Did not enter a valid argument type, please provide an integar, list, or numpy.ndarray')

    
    #--------------------------------------------------------------------    
    # Convert input temperature to K if supplied in Celsius (default), and 
    # convert to Celsius if supplied in K (keyword supplied)
    #--------------------------------------------------------------------
    if 'use_T_K' not in varargin.keys():
        Tk = T + 273.15
    else:
        Tk = T
        T = Tk-273.13        
        
    #--------------------------------------------------------------------
    # Convert pressure to Pa if supplied as hPa
    #--------------------------------------------------------------------        
    if 'use_pres_Pa' not in varargin.keys():
        p = p*100.

    #--------------------------------------------------------------------
    # Determine constants and reference values.
    #--------------------------------------------------------------------

    p_0 = 1.e5; # Pa
    R_d = 287.058; # gas constant for dry air [J/(kg*K)].
    R_v = 461.5; # gas constant for water vapor [J/(kg*K)].
    c_p = 1005.7; # +- 2.5 [J/(kg*K)] - specific heat capacity of dry air at 273K in a constant pressure.
    
    #-------------------------------------------------------------------- 
    # liquid water saturations polynom coefficients based on Flatau et al., 1992.  
    #--------------------------------------------------------------------
    p_water_s_coeff = [0.209339997e-15,\
        -0.373208410e-12,\
        0.892344772e-10,\
        0.196237241e-7,\
        0.305903558e-5,\
        0.264461437e-3,\
        0.143064234e-1,\
        0.444007856,\
        6.11213476]    
    
    #-----------------------------------------------------f---------------
    # ice water saturations polynom coefficients based on Flatau et al., 1992.
    #--------------------------------------------------------------------
    p_water_i_coeff = [0.262655803e-14,\
        0.149436277e-11,\
        0.387940929e-9,\
        0.602780717e-7,\
        0.614396778e-5,\
        0.420547422e-3,\
        0.188369801e-1,\
        0.503109514,\
        6.11123516]    
 

    #--------------------------------------------------------------------
    # Calculate latent heat release.
    #--------------------------------------------------------------------
    # latent heat of vaporization in J/Kg (equation is a good approximation for -25 - 40 C range). 
    # See Rogers and Yau (1987), Table 2.1.
    L_v = (2500.8 - 2.36 * T + 0.0016 * (T**2) - 0.00006 * (T**3)) * 1000 
    
    # latent heat of sublimation in J/Kg (almost constant between -40 - 0 C).
    #see Rogers and Yau (1987), Table 2.1.    
    L_s = (2834.1 - 0.29 * T - 0.004 * (T**2)) * 1000 
    
    #--------------------------------------------------------------------  
    # Calculate water vapor saturation pressure
    #-------------------------------------------------------------------- 
    if sat_pres_formula == 'Flatau': # Flatau et al. (1992)
        water_e_s = np.polyval(p_water_s_coeff, T) * 100 
        # valid for -100 to 0 C 
        # (~0.1% max error, T is in Celsius). 
        # Based on Flatau et al, J. Applied Met., 1992 (table 5, relative error norm)
        
        water_e_i = np.polyval(p_water_i_coeff, T) * 100 
        # valid for -75 to 0 C 
        # (~0.384% max error, T is in Celsius). 
        # Based on Flatau et al, J. Applied Met., 1992 (table 5, relative error norm)
        
    elif sat_pres_formula == 'Alduchov_Eskridge':
        water_e_s = 6.1094 * np.exp((17.625 * T) / (T + 243.04)) * 100 
        # valid for -40 - +50 C 
        #(0.384% max error, T is in Celsius). This equation is empirical based on 
        # Alduchov and Eskridge, J. Applied Met., 1996
        
        water_e_i = 6.1121 * np.exp((22.587 * T) / (T + 273.16)) * 100; 
        # valid for -80 - 0 C 
        # (0.414% max error, T is in Celsius). This equation is empirical based on 
        # Alduchov and Eskridge, J. Applied Met., 1996
        
    elif sat_pres_formula == 'Murphy_Koop':
        water_e_s = np.exp(54.842763 - 6763.22/Tk - 4.210*np.log(Tk) + \
                           0.000367*Tk + np.tanh(0.0415*(Tk-218.8)) * \
                           (53.878 - 1331.22/Tk - 9.44523*np.log(Tk) + 0.014025*Tk))
        # Murphy and Koop (2005)
        # valid for 123 < T < 332 K
       
        water_e_i = np.exp(9.550426 - 5723.265/Tk + 3.53068*np.log(Tk) - 0.00728332*Tk)
        # valid for T > 110 K.
        # Murphy and Koop (2005)
        
    elif sat_pres_formula == 'Teten':
        # Teten's formula (ERA-Interim)
        # valid for -40 - +50 C 
        # (0.384% max error, T is in Celsius). This equation is empirical based on 
        # Alduchov and Eskridge, J. Applied Met., 1996       
        water_e_s = 6.1121 * np.exp((17.502 * T) / (T + 240.97)) * 100
        
        water_e_i = 6.1121 * np.exp((22.587 * T) / (T + 273.86)) * 100
        # Teten's formula (ERA-Interim)  
        # valid for -80 - 0 C 
        # (0.414% max error, T is in Celsius). This equation is empirical based on 
        # Alduchov and Eskridge, J. Applied Met., 1996        
        
    elif sat_pres_formula == 'Emmanuel':
        water_e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5)) * 100 
        #Emanuel (1994) (AMPS)
        # valid for -40 - +50 C 
        #(0.384% max error, T is in Celsius). This equation is empirical based on 
        # Alduchov and Eskridge, J. Applied Met., 1996
        #Emanuel (1994) (AMPS)
    
        water_e_i = np.exp(23.33086 - 6111.72784 / Tk + 0.15215 * np.log(Tk)) * 100
        #Emanuel (1994) (AMPS)
        # valid for -80 - 0 C 
        # (0.414% max error, T is in Celsius). This equation is empirical based on 
        #Alduchov and Eskridge, J. Applied Met., 1996
    else:
        #We will default to using Flatau if not specified
        
        water_e_s = np.polyval(p_water_s_coeff, T) * 100 
        # valid for -100 to 0 C 
        # (~0.1% max error, T is in Celsius). 
        # Based on Flatau et al, J. Applied Met., 1992 (table 5, relative error norm)
        
        water_e_i = np.polyval(p_water_i_coeff, T) * 100 
        # valid for -75 to 0 C 
        # (~0.384% max error, T is in Celsius). 
        # Based on Flatau et al, J. Applied Met., 1992 (table 5, relative error norm)
            
    
    #--------------------------------------------------------------------      
    # Calculate water vapor pressure and mixing ratio.
    #--------------------------------------------------------------------  
    result = True if "w" in varargin.keys() else False
    if result == True:    
        water_e = water_w * p / (0.62197 + water_w) # in Pa
        RH = water_e / water_e_s * 100
    else:
        water_e = RH * water_e_s  / 100 # get water vapor pressure from 
        #the RH and saturation water pressure. 
        #see: http://www.srh.noaa.gov/images/epz/wxcalc/mixingRatio.pdf.
        water_w = 0.62197 * water_e / (p - water_e) # mixing ratio in g/kg. 
        #see: http://www.srh.noaa.gov/images/epz/wxcalc/mixingRatio.pdf.

    #--------------------------------------------------------------------      
    # Calculate potential temperature and virtual temperature
    #--------------------------------------------------------------------      
    p_d = p - water_e # Dry air partial pressure.
    Tv = Tk*(1 + (water_w*1e-3)/0.62197)/(1 + (water_w*1e-3))
    Theta = Tk*(p_0/p)**(R_d/c_p); # potential temperature.
    
    #--------------------------------------------------------------------      
    # Calculate equivalent potential temperature.
    #--------------------------------------------------------------------          
    Theta_e = Tk* (p_0 / p_d)**(R_d/ c_p) * (RH/ 100)**((water_w)*R_v/c_p)*np.exp(L_v*(water_w)/(c_p*Tk))
    # equivalent potential temperature (neglecting r*c, i.e., total water mixing ratio and heat 
    # capacity of liquid water, while retaining good accuracy). 
    # see: http://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
    
    #--------------------------------------------------------------------      
    # Calculate virtual potential temperature.
    #--------------------------------------------------------------------     
    Theta_v = Tv*(p_0/ p)**(R_d/ c_p) # virtual potential temperature.

    #--------------------------------------------------------------------      
    # Calculate first-order equivalent potential temperature error.
    #--------------------------------------------------------------------     
    result = True if "theta_e_un" in varargin.keys() else False
    if result == True:
        water_e_un = delta_RH * water_e_s / 100      
        delta_water_w = 0.62197* p / (water_e - p)**2 * water_e_un
        Theta_e_un =  np.sqrt((((p_0 / p_d)**(R_d/ c_p) * (RH/ 100)**((water_w)* R_v / c_p) * \
                               (Tk - L_v* (water_w)/ c_p)* np.exp(L_v* (water_w)/ (c_p*Tk)) / Tk) * \
                               delta_Tk)**2 + (Tk* (p_0 / p_d)**(R_d/ c_p)*(RH/ 100)**((water_w)* R_v / c_p) * \
                               np.exp(L_v* (water_w)/ (c_p*Tk))* ((water_w)* R_v / c_p)/ RH * delta_RH)**2 + \
                               (Tk* (p_0 / p_d)**(R_d/ c_p) * (RH/ 100)**((water_w)* R_v / c_p) * exp(L_v* (water_w)/ \
                               (c_p*Tk))* ((L_v/ (c_p*Tk)) + np.log(RH/ 100)* (R_v / c_p)) * delta_water_w)**2 )

    #--------------------------------------------------------------------  
    # Calculate RH_above ice and liquid water
    #--------------------------------------------------------------------           
    RH_i = water_e / water_e_i * 100
    RH_l = water_e / water_e_s * 100

    #--------------------------------------------------------------------           
    # Arranging ouput in a single dictionary
    #-------------------------------------------------------------------- 
    var_full_names = {'L_v':'latent heat of vaporation',\
                      'L_s':'latent heat of sublimation',\
                      'e_s':'saturation vapor pressure over water',\
                      'e_i':'saturation vapor pressure over ice',\
                      'w':'water vapor mixing ratio',\
                      'e':'water vapor pressure',\
                      'T_v':'virtual temperature',\
                      'Theta':'potential temperature',\
                      'Theta_e':'equivalent potential temperautre',\
                      'Theta_v':'virtual potential temperature',\
                      'RH_i':'relative humidity over ice',\
                      'RH_l':'relative humidity over liquid water',\
                      'q':'specific humidity',\
                      'w_s':'saturation water vapor mixing ratio',\
                     }
    
    var_units = {'L_v':'J/(kg*K)',\
                      'L_s':'J/(kg*K)',\
                      'e_s':'Pa',\
                      'e_i':'Pa',\
                      'e':'Pa',\
                      'w':'g/kg',\
                      'T_v':'K',\
                      'Theta':'K',\
                      'Theta_e':'K',\
                      'Theta_v':'K',\
                      'RH_i':'%',\
                      'RH_l':'%',\
                      'q':'g/kg',\
                      'w_s':'g/kg',\
                     }
    
    
    # Separate dictionary if theta_e uncertainties are calculated
    result = True if "theta_e_un" in varargin.keys() else False
    if result == True:  
        output_dict = {'L_v':L_v,\
                       'L_s':L_s,\
                       'e_s':water_e_s,\
                       'e_i':water_e_i,\
                       'e':water_e,\
                       'w':water_w*1.e3,\
                       'T_v':Tv,\
                       'Theta':Theta,\
                       'Theta_e':Theta_e,\
                       'Theta_v':Theta_v,\
                       'RH_i':RH_i,\
                       'RH_l':RH_l,\
                       'q':water_w/(1+water_w)*1.e3,\
                       'w_s':0.622*water_e_s/(p - water_e_s)*1.e3,\
                       'water_e_un':delta_water_w*1e3,\
                       'Theta_e_un':Theta_e_un,\
                       'var_full_names':var_full_names,\
                       'var_units':var_units,\
                      }  
    else:
        output_dict = {'L_v':L_v,\
                       'L_s':L_s,\
                       'e_s':water_e_s,\
                       'e_i':water_e_i,\
                       'e':water_e,\
                       'w':water_w*1.e3,\
                       'T_v':Tv,\
                       'Theta':Theta,\
                       'Theta_e':Theta_e,\
                       'Theta_v':Theta_v,\
                       'RH_i':RH_i,\
                       'RH_l':RH_l,\
                       'q':water_w/(1+water_w)*1.e3,\
                       'w_s':0.622*water_e_s/(p - water_e_s)*1.e3,\
                       'var_full_names':var_full_names,\
                       'var_units':var_units,\
                      }
        
    # returns a dictionary named "output_dict" that is equivalent to Matlab's Output_struct.*
    return output_dict
     
    
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TESTING
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!


#T = [0,-10]
#RH = [100,100]
#p = [1000,900]
##w = 10. # passed in g/kg
#tmp1 = calculate_theta_and_more(T,p,RH=RH,sat_pres_formula='Flatau')
#
#for key,val in tmp1.items():
#    if (key != 'var_full_names') and (key != 'var_units'):
#        print(key,type(val),val)
#        
#print(aaaaa)

