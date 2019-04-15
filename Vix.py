# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:34:10 2019

@author: s3179575
"""

import os
os.chdir('X:\\My Documents\\LPR\\LPR')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




vix = pd.read_table("vixcurrent.csv",sep=",")

vix_h = pd.read_excel('vixarchive.xls', index_col=0)






