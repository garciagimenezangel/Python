# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:08:26 2020

@author: angel.gimenez
"""
import dill   
       
# file    
from os.path import expanduser
home = expanduser("~")            
filename = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\gdbLayerZ30data.pkl'

# save session 
dill.dump_session(filename)

# load session
dill.load_session(filename)


