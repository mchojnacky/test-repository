# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:04:33 2020

@author: michalc
"""
import os
os.chdir('C:\Python\E1')

import glob
import pandas as pd
import numpy as np
#import scipy
from scipy import signal
import matplotlib.pyplot as plt



#make a list of filenames
hi_names = glob.glob('C:\Python\E1\hi\*')

lo_names = glob.glob("C:\Python\E1\lo\*")

C = 1*10**-6
R = 330
#new df with f, Vout/Vin and calculated ratio

L = []
for file in hi_names:
    tst = pd.read_csv(file)
    f = float(os.path.basename(file).rstrip('.csv'))
    tst.rename(columns={" Ch 0 (volts)": "ch0", " Ch 1 (volts)": "ch1"}, inplace=True)
    
    max0 = max(tst.ch0)
    min0 = min(tst.ch0)
    max1 = max(tst.ch1)
    min1 = min(tst.ch1)
    Vout = (max0 - min0)/2
    Vin = (max1 - min1)/2
    
    peakarr0, _ = signal.find_peaks(tst['ch0'], prominence = .9*max0, distance=len(tst)/12)
    peakarr1, _ = signal.find_peaks(tst['ch1'], prominence = .9*max1, distance=len(tst)/12)
    
    l0 = len(peakarr0)
    l1 = len(peakarr1)
    tpeak0 = tst.iloc[peakarr0,0].to_numpy()
    tpeak1 = tst.iloc[peakarr1,0].to_numpy()
    min_size = min(tpeak0.size, tpeak1.size)
    shift = tpeak1[:min_size] - tpeak0[:min_size] 
    avg_shift = np.mean(shift)
  
    dat = [f, Vin, Vout, avg_shift, l0, l1]
    L.append(dat)

hi_data = pd.DataFrame(L, columns = ['frequency (Hz)', 'Vin (V)', 'Vout (V)', 'time shift (ms)', 'Vout peaks', 'Vin peaks'])
hi_data = hi_data.sort_values('frequency (Hz)').reset_index(drop = True)
hi_data.to_csv('hi_data.csv')

#create a data frame for summarized data, columns = ['frequency (Hz)', 'Vout/Vin', 'Vout/Vin calc']
# and columns = ['frequency (Hz)', 'time shift (ms)']
#plot summarized data


hi_data['Vout/Vin'] = hi_data['Vout (V)']/hi_data['Vin (V)']
#create a summary data frame from a copy of the original to avoid SettingWithCopy warning
hi_V = hi_data[['frequency (Hz)', 'Vout/Vin']].copy()
hi_V['Vout/Vin calc'] = 2*np.pi*hi_V['frequency (Hz)']*C*R/(np.sqrt((2*np.pi*hi_V['frequency (Hz)']*C*R)**2+1))
hi_V.to_csv('hi_V.csv')

# plt.plot('frequency (Hz)', 'Vout/Vin', data=hi_V, marker='D', markerfacecolor='None', markeredgecolor ='b', markersize=6, linestyle='None', label = 'measured')
# plt.plot('frequency (Hz)', 'Vout/Vin calc', data=hi_V, #marker='o', markerfacecolor='None', markeredgecolor = 'magenta', markersize = 6, linestyle='None')
#          color = 'magenta', linewidth =1, label='calculated')
# plt.xlabel('frequency (Hz)')
# plt.ylabel('signal ratio (Vout/Vin)')
# plt.legend()
# plt.yscale('log')
# plt.xscale('log')

# hi_t = hi_data[['frequency (Hz)', 'time shift (ms)']].copy()
# print(hi_t)

# plt.plot('frequency (Hz)', 'time shift (ms)', data=hi_t, marker='o', markerfacecolor='None', markeredgecolor ='r', markersize=6, linestyle='None')
# plt.xlabel('frequency (Hz)')
# plt.ylabel('time shift (ms)')
# plt.yscale('log')
# plt.xscale('log')
         
M = []
for file in lo_names:
    tst = pd.read_csv(file)
    f = float(os.path.basename(file).rstrip('.csv'))
    tst.rename(columns={" Ch 0 (volts)": "ch0", " Ch 1 (volts)": "ch1"}, inplace=True)
    
    max0 = max(tst.ch0)
    min0 = min(tst.ch0)
    max1 = max(tst.ch1)
    min1 = min(tst.ch1)
    Vout = (max0 - min0)/2
    Vin = (max1 - min1)/2
    
    peakarr0, _ = signal.find_peaks(tst['ch0'], prominence = .9*Vout, distance=len(tst)/12)
    peakarr1, _ = signal.find_peaks(tst['ch1'], prominence = .9*Vin, distance=len(tst)/12)
    
    l0 = len(peakarr0)
    l1 = len(peakarr1)
    tpeak0 = tst.iloc[peakarr0,0].to_numpy()
    tpeak1 = tst.iloc[peakarr1,0].to_numpy()
    min_size = min(tpeak0.size, tpeak1.size)
    shift = tpeak1[:min_size] - tpeak0[:min_size] 
    avg_shift = np.mean(shift)
  
    dat = [f, Vin, Vout, avg_shift, l0, l1]
    M.append(dat)

lo_data = pd.DataFrame(M, columns = ['frequency (Hz)', 'Vin (V)', 'Vout (V)', 'time shift (ms)', 'Vout peaks', 'Vin peaks'])
lo_data = lo_data.sort_values('frequency (Hz)').reset_index(drop = True)
lo_data.to_csv('lo_data.csv')

lo_data['Vout/Vin'] = lo_data['Vout (V)']/lo_data['Vin (V)']
#create a summary data frame from a copy of the original to avoid SettingWithCopy warning
lo_V = lo_data[['frequency (Hz)', 'Vout/Vin']].copy()
lo_V['Vout/Vin calc'] = 1/(np.sqrt((2*np.pi*lo_V['frequency (Hz)']*C*R)**2+1))
lo_V.to_csv('lo_V.csv')

lo_t = lo_data[['frequency (Hz)', 'time shift (ms)']].copy()
print(lo_t)

# plt.plot('frequency (Hz)', 'time shift (ms)', data=lo_t, marker='o', markerfacecolor='None', markeredgecolor ='b', markersize=6, linestyle='None')
# plt.xlabel('frequency (Hz)')
# plt.ylabel('time shift (ms)')
# plt.yscale('symlog')
# plt.xscale('log')

# plt.plot('frequency (Hz)', 'Vout/Vin', data=lo_V, marker='D', markerfacecolor='None', markeredgecolor ='purple', markersize=6, linestyle='None', label = 'measured')
# plt.plot('frequency (Hz)', 'Vout/Vin calc', data=lo_V, #marker='o', markerfacecolor='None', markeredgecolor = 'magenta', markersize = 6, linestyle='None')
#          color = 'green', linewidth =1, label='calculated')
# plt.xlabel('frequency (Hz)')
# plt.ylabel('signal ratio (Vout/Vin)')
# plt.legend()
# plt.yscale('log')
# plt.xscale('log')

# tst = pd.read_csv('lo//10000.csv')

# tst.rename(columns={" Ch 0 (volts)": "ch0", " Ch 1 (volts)": "ch1"}, inplace=True)

# max0 = max(tst.ch0)
# min0 = min(tst.ch0)
# max1 = max(tst.ch1)
# min1 = min(tst.ch1)

# Vout = (max0 - min0)/2
# Vin = (max1 - min1)/2


# #distance is number of samples... may need to change
# peakarr0, _ = signal.find_peaks(tst['ch0'], prominence = .9*Vout, distance=len(tst)/12)
# peakarr1, _ = signal.find_peaks(tst['ch1'], prominence = .9*Vin, distance=len(tst)/12)


# tpeak0 = tst.iloc[peakarr0,0].to_numpy()
# tpeak1 = tst.iloc[peakarr1,0].to_numpy()

# min_size = min(tpeak0.size, tpeak1.size)
# shift = tpeak1[:min_size] - tpeak0[:min_size] 
# avg_shift = np.mean(shift)

# dat = [[Vin, Vout, avg_shift]]