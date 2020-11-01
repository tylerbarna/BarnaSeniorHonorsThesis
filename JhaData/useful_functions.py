import csv
import dis
import inspect
import os
import sys

import astropy
import astroquery
import eleanor
#import tess_stars2px ## Currently unnecessary
import lightkurve as lk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import sncosmo
import time
import warnings
warnings.filterwarnings('ignore')

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.timeseries import TimeSeries
from astropy.visualization import time_support
time_support()
from astropy.visualization import quantity_support
quantity_support()

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import linalg as la
from scipy import optimize
from scipy import integrate
from scipy import stats

from IPython.display import display_html
from IPython.display import Image

def mkdir(directory): ## creates a directory if it doesn't exist
    ## credit to https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def display_side_by_side(*args): ##displays pandas DataFrames side by side
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def unravel(list): ## creates an array from a list of arrays
    return np.array([i for array in list for i in array])

def interpToMatch(item1,item2):
    ## Function takes two pandas DataFrames with a 'time' column (float or integer type)
    ## and interpolates the data to match the set with a smaller number of points
    ## with the interpolated DataFrames being returned as a tuple
    item1_indexed = item1.set_index('time')
    item2_indexed = item2.set_index('time')
    #display(item1_indexed)
    item1_length = len(item1_indexed.index)
    item2_length = len(item2_indexed.index)
    #display(item1_length)
    if item1_length >= item2_length:
        minun = item2_indexed.index.min()
        plusle = item2_indexed.index.max()
        numPoints = item2_length
    elif item1_length <= item2_length:
        minun = item1_indexed.index.min()
        plusle = item1_indexed.index.max()
        numPoints = item1_length
    #display(minun)
    #display(plusle)
    
    #numPoints = abs(plusle-minun)
    newIndex = np.linspace(minun,plusle-1,numPoints)
    #display(numPoints)
    #display(newIndex)
    
    item1_interp = pd.DataFrame(index=newIndex)
    item1_interp.index.name = item1_indexed.index.name
    item2_interp = pd.DataFrame(index=newIndex)
    item2_interp.index.name = item2_indexed.index.name

    for colname, col in item1_indexed.iteritems():
        item1_interp[colname] = np.interp(newIndex,item1_indexed.index,col)
    for colname, col in item2_indexed.iteritems():
        item2_interp[colname] = np.interp(newIndex,item2_indexed.index,col)
    item1_interp.reset_index(inplace=True)
    item2_interp.reset_index(inplace=True)
    
    return item1_interp, item2_interp

def interpToData(data,data_index='time',arg_index='time',*args):
    ## More generalized version of interpToMatch(). Takes an argument for a reference
    ## DataFrame and a variable number of DataFrames to be interpolated so that
    ## they match the time sampling of the reference DataFrame. Like interpToMatch(),
    ## DataFrames must have a 'time' column of an integer or float type.
    ## Function returns an array containing the reference DataFrame as the first
    ## item followed by the interpolated DataFrames in the order in which they were
    ## passed to the function
    interpArray = []
    interpArray.append(data)
    
    data_indexed = data.set_index(str(data_index))
    data_length = len(data_indexed.index)
    minun = data_indexed.index.min()
    plusle = data_indexed.index.max()
    newIndex = data_indexed.index
    
    for arg in args:
        arg_indexed = arg.set_index(str(arg_index))
        arg_interp = pd.DataFrame(index=newIndex)
        arg_interp.index.name = arg_indexed.index.name
        for colname, col in arg_indexed.iteritems():
            arg_interp[colname] = np.interp(newIndex,arg_indexed.index,col)
        arg_interp.reset_index(inplace=True)
        interpArray.append(arg_interp)
    return interpArray

def lcImport(directory, head=None):
    ## This is a workaround that imports the lightcurves from Fausnaugh correctly; due to a quirk
    ## with his formatting, pandas initially sets the BTJD values to be the index and shifts the
    ## rest of the columns to the left by one.
    
    if head: ## in case there's something non-standard or want to change col names, don't expect to use this
        header = head
    elif not head: ## Fausnaugh data columns
        header = ['BTJD', 'TJD', 'cts', 'e_cts', 'bkg', 'bkg_model', 'bkg2', 'e_bkg2'] 
    
    snPandas = pd.read_csv(directory,delim_whitespace=True, header=0, names=header)
    snPandas = snPandas.shift(1,axis=1) ## Fixes issue with data shifted one col to the left
    snPandas[header[0]] = snPandas.index ## Populates BTJD col with correct data
    snPandas = snPandas.reset_index(drop=True) ## Resets index to be standard
    
    return snPandas