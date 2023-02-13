#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# give_me_files_and_subfolders.py
# Original creator: Israel Silber
# Converted from Matlab to Python by McKenna Stanford
# Author: McKenna W. Stanford (mws2175@columbia.edu)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#-------------------------------------------------------
# Python Imports
#-------------------------------------------------------
import numpy as np
import glob
import inspect
from file_struct import file_struct as fs
import os

#-------------------------------------------------------
# Keyword arguments (kwargs):
#    file_name
#      - allow user to supply a filename
#-------------------------------------------------------


#----------------------------------------------
# Creating a dummy class that is the equivalent
# of a "struct" in Matlab. The end result will
# be a Python list (of files) with attributes
# including the file name (fname), the file path
# (fpath), and the file size (fsize)
#****
# NOTE: class for "file_struct" is read in now
#----------------------------------------------
if False:
    class file_struct:
        def __init__(self, fname, fpath, fsize):
            self.fname = fname
            self.fpath = fpath
            self.fsize = fsize        

    tmp = []
    for ii in range(10):
        tmp.append(fs('fname_'+str(ii),'fpath_'+str(ii),'fsize_'+str(ii)))


#---------------------------------------
# Simple kernel to get the attributes of
# the file_struct class and print them
# to screen. 
#---------------------------------------
if False:
    # getmembers() returns all the  
    # members of an object  
    for i in inspect.getmembers(tmp[0]):

        # to remove private and protected 
        # functions 
        if not i[0].startswith('_'): 

            # To remove other methods that 
            # doesnot start with a underscore 
            if not inspect.ismethod(i[1]):  
                print(i)         
#---------------------------------------
#---------------------------------------



def give_me_files_and_subfolders(Search_String,Path,**kwargs):
    """ 
    This function searches a parent Path (must be provided)
    via one or more search strings. The search strings must
    be provided. If desire is to list all files/folders
    within the parent directory, simply supply ''. This function
    will search all subfolders recursively nested within the
    parenth Path.
    
    The search string can be provided as a string, list,
    or tuple. Program will fail if one of these 3 types
    are not provided.
    
    This function utilizes a class called "file_struct"
    in which a file_struct object contains the attributes "fname, 
    fpath, and fsize", corresponding to the file name, file path,
    and file size (in KB), respectively. Practically, the file_struct
    objects are elements of the list returned by the function (flist_struct).
    
    There is one acceptable keyword argument named "flist_struct_input".
    This can be passed if the user desires to append to an existing (and supplied)
    list of file_struct objects. However, most commonly this keyword
    argument is used for the recursion capabilities and will not be needed
    by the user.
    
    Output is a Python list of file_struct objects nested withint the supplied
    parent Path and matching the supplied search string(s).
    
    """
    varargin=kwargs
    
    # The user can supply a file list with 'file_struct' class objects,
    # but this conditional is mainly for recursion, in which recursive
    # entries will already have a 'flist_struct' which will be passed
    # in as a keyword argument
    
    if 'flist_struct_input' in varargin.keys():
        flist_struct = varargin['flist_struct_input']
    else:
        flist_struct = []
        
    #====================================================
    # In the case of multiple search strings,
    # we want the final type to be a tuple. So checking
    # the argument type and converting to a tuple if
    # supplied as a string or list, even if it is only one
    # entry, in which case the loop below will just loop
    # once.
    #====================================================

    arg_type = type(Search_String)
    #print(arg_type)
    if arg_type is str:
        #print('arg type is str, converting to tuple')
        supplied_search_string = Search_String
        # need to make it a list first, hence square brackets
        tmp_list = [supplied_search_string]
        Search_String = tuple(tmp_list)
    elif arg_type is list:
        #print('arg type is list, converting to tuple')
        supplied_search_string = Search_String
        Search_String = tuple(supplied_search_string)
    elif arg_type is tuple:
        #print('arg type is tuple')
        pass
    else:
        raise RuntimeError('invalid argument type supplied as "Search String". Please provide a string, list, or tuple')
        

    #-------------------------------------
    # Loop through elements of tuple
    # containing one or more search
    # strings
    #-------------------------------------
    for jj in range(len(Search_String)):
        tmp_list = glob.glob(Path+'*'+Search_String[jj]+'*') #searching files with a matching string in the given path.
        tmp_list = sorted(tmp_list)
        #print(tmp_list)
        if len(tmp_list) > 0:
            # loop through files
            for ii in range(len(tmp_list)):
                # check to see if element of list is a directory or a file
                isdir = os.path.isdir(tmp_list[ii])

                # if element in list is a file, save the path, file name, and file size
                # as attributes of the file_struct class. File size is given in KB
                if isdir is False:
                    file_path = os.path.dirname(tmp_list[ii])+'/'
                    file_name = tmp_list[ii]
                    path_split = str.split(tmp_list[ii],'/')
                    file_name = path_split[-1]
                    file_size = os.stat(tmp_list[ii]).st_size/1.e3 # convert from Bytes to KB
                    
                    # finally, append the flist_struct list with the file_struct
                    # class object containing the file_path, the file_name, and 
                    # the file_size
                    flist_struct.append(fs(file_name,file_path,file_size))
                    
                # if a directory, skip ##Now commented out
                #elif isdir is True:
                #    file_path = os.path.dirname(tmp_list[ii])+'/'
                #   # print(file_path) 
                
    #-------------------------------------------------------------          
    # Recursion: Now look through all subfolders using the same
    # search string. This operates by going into ALL subfolders
    # nested within the parent Path and performing the function
    # again.
    #-------------------------------------------------------------
    tmp_list = glob.glob(Path+'*')
    tmp_list = sorted(tmp_list)
    tmp_list_dir = []
    for entry in tmp_list:
        isdir = os.path.isdir(entry)
        if isdir is True:
            tmp_list_dir.append(entry)

    if len(tmp_list_dir) > 0.:
        for ii in range(len(tmp_list_dir)):
            New_Path = tmp_list_dir[ii]+'/'
            give_me_files_and_subfolders(Search_String,New_Path,\
                                         flist_struct_input=flist_struct)
            
                    
        
    return flist_struct

        
        
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TESTING
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    

#search_string = 'here I am'
#search_string = ['here','i','am']
#search_string = ('here','i','am')
#search_string = {'here','i','am'}

#Path = '/Users/mckennastanford/Documents/'
#search_string = 'nasa'
#search_string = ('nasa_1','nasa_2')


#final_list = give_me_files_and_subfolders(search_string,Path)
#print(tmp)

#for tmp in final_list:
#    print(tmp.fname)
#    print(tmp.fpath)
#    print(tmp.fsize)
#    print('')