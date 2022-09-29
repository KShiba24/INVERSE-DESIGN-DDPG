"""
execute this program after execute train.py
this program use Gaussian16.
if you execute this program, computer installed gaussian16 is used. 
"""

import math
import numpy as np
import pandas as pd
import csv
from package.enviroment_validation import Env
import matplotlib.pyplot as plt

###hyper parameter ##############
total_atom_num = 12
inverse_atom_num = 1
###########################################
env = Env('Cd6Se6/Cd6Se6_abs.csv',total_atom_num, inverse_atom_num)
target_spectra = pd.read_csv('Cd6Se6/Cd6Se6_abs.csv')[:]['label'].values
target_peek_wavelength = np.argmax(target_spectra)*1.6
df = np.array(pd.read_csv('validation_state.csv',header=None))
final_position = df[-300:]
print(final_position)
predict_spectra = []
for i in range(len(final_position)):
    particle = final_position[i]
    intens = env.step(particle)
    intens = intens.reshape((1,500))
    predict_spectra.append(intens[0])
    with open('validation_spectra.csv', 'a') as f: 
        writer = csv.writer(f)
        writer.writerow(intens[0])
target_peek_wavelength_list = []
predict_peek_wavelength = []
for i in range(len(predict_spectra)):
    target_peek_wavelength_list.append(target_peek_wavelength)
    predict_peek_wavelength.append(np.argmax(predict_spectra[i])*1.6)


# Fig.6 of jounarl 
fig1 = plt.figure()
ax = fig1.add_subplot()
ax.scatter(target_peek_wavelength_list, predict_peek_wavelength)
ax.axhline(y=target_peek_wavelength, xmin=0.05, xmax=0.95,
            color='red',
            lw=2,
            ls='--',
            alpha=0.6)
ax.set_xlim(300,450)
ax.set_ylim(300,800)
ax.set_xlabel('target peak wavelength [nm]')
ax.set_ylabel('predict peak wavelength [nm]')
plt.show()
