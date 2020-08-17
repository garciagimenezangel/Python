# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:06:09 2020

@author: angel.gimenez
"""

import sys
from os.path import expanduser
home = expanduser("~")
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import functions

for x in functions.allFiles("C:\\Users\\angel.gimenez\\Documents\\DATA\\OBServ\\ESYRCE\\Esyrce1992_2000\\", "dbf"):
    print("Processing..."+x)
    functions.dbf_to_csv(x)