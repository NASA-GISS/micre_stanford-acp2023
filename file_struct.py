#----------------------------------------------
# Creating a dummy class that is the equivalent
# of a "struct" in Matlab. The end result will
# be a Python list (of files) with attributes
# including the file name (fname), the file path
# (fpath), and the file size (fsize)
# Author: McKenna W. Stanford (mws2175@columbia.edu)
#----------------------------------------------
class file_struct(object):
    """ Dummy class that contains a file name (fname),
    a file path (fpath) and a file size in KB (fsize) """
    def __init__(self, fname, fpath, fsize):
        self.fname = fname
        self.fpath = fpath
        self.fsize = fsize   
