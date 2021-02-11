# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:06:58 2021

@author: angel
"""
def isFallow(dataRow):
    try:
        d5cul = dataRow.D5_CUL
        d5cul = d5cul[0:2]
        return (d5cul == 'BA')
    except:
        return False

data['isFallow'] = [isFallow(data.loc[i]) for i in data.index]
databarb = data.where(data['isFallow'])
databarb = databarb.dropna(thresh=1)
