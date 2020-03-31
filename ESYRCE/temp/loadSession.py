# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:08:26 2020

@author: angel.gimenez
"""
import dill   
       
# file                 
filename = '.\\Documents\\REPOSITORIES\\Python\\ESYRCE\\gdbLayerZ30_LoadedSession.pkl'

# save session 
dill.dump_session(filename)

# load session
dill.load_session(filename)


