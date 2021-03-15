# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:35:05 2020

@author: michalc
"""
import os
os.chdir('C:\Python\E1')

import glob
import pandas as pd
import numpy as np
import scipy
from scipy import signal

#make a list of filenames
hi_names = glob.glob('C:\Python\E1\hi\*')

lo_names = glob.glob("C:\Python\E1\lo\*")


hi_dfs = [pd.read_csv(filename) for filename in hi_names]

for dataframe, filename in zip(hi_dfs, hi_names):
    dataframe['frequency (Hz)'] = filename
    
#hi_df = pd.concat(hi_dfs)
#hi_df['frequency (Hz)'] = hi_df['frequency (Hz)'].str.split('\\').str[-1].str.rstrip('.csv')

#print(hi_df)

# lo_dfs = [pd.read_csv(filename) for filename in lo_names]

# for dataframe, filename in zip(lo_dfs, lo_names):
#     dataframe['frequency (Hz)'] = filename
    
# lo_df = pd.concat(lo_dfs)
# lo_df['frequency (Hz)'] = lo_df['frequency (Hz)'].str.split('\\').str[-1].str.rstrip('.csv')

# i1 = hi_dfs[0][['Time (ms)', ' Ch 0 (volts)']]
# i2 = hi_dfs[0][['Time (ms)', ' Ch 1 (volts)']]

# corr = scipy.signal.correlate(i1, i2)

# def lag_finder(i1, i2, sr):
#     n = len(i1)

#     corr = signal.correlate(i2, i1, mode='same') / n

#     delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
#     delay = delay_arr[np.argmax(corr)]
#     print('i2 is ' + str(delay) + ' behind i1')

#     plt.figure()
#     plt.plot(delay_arr, corr)
#     plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
#     plt.xlabel('Lag')
#     plt.ylabel('Correlation coeff')
#     plt.show()

#need to deal with Seconds vs Milliseconds data - during import?

tst = pd.read_csv('hi//150.csv')

i0 = tst[['Time (ms)',' Ch 0 (volts)']]
i0 = i0.rename(columns={" Ch 0 (volts)": "ch0"})
i1 = tst[['Time (ms)',' Ch 1 (volts)']]
i1 = i1.rename(columns={" Ch 1 (volts)": "ch1"})

#need to find a way to determine prominence parameters
peakarr0, _ = signal.find_peaks(i0['ch0'], prominence = 1, distance=100)
peakarr1, _ = signal.find_peaks(i1['ch0'], prominence = 3, distance=100)


tpeak0 = i0.iloc[peakarr0,0]
tpeak0.drop(i0.iloc[peakarr0,0].tail(1).index, inplace=True)
tpeak1 = i1.iloc[peakarr1,0]

shift = tpeak1.to_numpy() - tpeak0.to_numpy()
np.mean(shift)

# sr = len(i1)

# lag_finder(i1, i2, sr)

#i1['max'] = i1.ch0[(i1.ch0.shift(1) < i1.ch0) & (i1.ch0.shift(-1) < i1.ch0)]