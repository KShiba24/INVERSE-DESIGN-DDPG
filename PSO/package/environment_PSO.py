import sys
sys.path.append("../package")
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn.functional
import torch.utils.data
from torch.autograd import Variable
import cloudpickle
from RCM_all_pattern import Coulomb_matrix_all

cm_all = Coulomb_matrix_all()


class Env():
    def __init__(self, model_path, label_path, total_atom_num, inverse_atom_num):
        self.total_atom_num = total_atom_num
        self.inverse_atom_num = inverse_atom_num
        with open(model_path, 'rb') as f:
            self.model = cloudpickle.load(f)
        df = pd.read_csv(label_path)
        self.label_x = np.array(df['X'][total_atom_num-inverse_atom_num:total_atom_num])
        self.label_y = np.array(df['Y'][total_atom_num-inverse_atom_num:total_atom_num])
        self.label_z = np.array(df['Z'][total_atom_num-inverse_atom_num:total_atom_num])
        self.label = np.array(df['label'])
        self.x = df['X'][0:total_atom_num-inverse_atom_num]
        self.y = df['Y'][0:total_atom_num-inverse_atom_num]
        self.z = df['Z'][0:total_atom_num-inverse_atom_num]
        self.a_num = df['atomnum'][0:total_atom_num]

    def predict(self, input):
        input = torch.tensor(np.array(input.astype('f')))
        output = self.model(input)
        output =  output.detach().numpy().copy()
        return output
    
    def reset(self):
        total_reward = [0]
        self.t = 0
        a = np.random.uniform(-self.search_region,self.search_region,(self.inverse_atom_num, 3))
        for i in range(len(self.label_x)):
            a[i][0] +=self.label_x[i]
            a[i][1] +=self.label_y[i]
            a[i][2] +=self.label_z[i]
        return a.reshape((1,-1))
    

    def amount_of_change(self, shita, fai):
        dx = self.r * math.sin(shita) * math.cos(fai)
        dy = self.r * math.sin(shita) * math.sin(fai)
        dz = self.r * math.cos(shita)
        ds = [dx, dy, dz]
        return ds

    def mse_error(self, predict, label):
        loss = np.mean((predict - label)**2)
        return loss

    def step(self, state):
        p = cm_all.make_input_data(state, self.a_num,self.x,self.y,self.z)
        cm = cm_all.make_matrix(p)/1000
        q = self.predict(cm)
        loss = self.mse_error(q, self.label/1000)*10
        return loss
