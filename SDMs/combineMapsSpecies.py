# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:25:58 2020

@author: angel.gimenez
"""


import os
home = os.path.expanduser("~")

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if fullPath.endswith(".tif"):
                allFiles.append(fullPath)
                
    return allFiles

dirName = home + '\\Documents\\DATA\\OBServ\\SDMs\\';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
