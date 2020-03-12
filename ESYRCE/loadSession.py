# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:08:26 2020

@author: angel.gimenez
"""

import dill                            #pip install dill --user
filename = 'globalsave.pkl'
dill.dump_session(filename)

# and to load the session again:
dill.load_session(filename)

