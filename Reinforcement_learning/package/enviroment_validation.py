import math
import numpy as np
import pandas as pd
import csv
import os 
import subprocess
from subprocess import Popen



class Env():
    def __init__(self, label_path,total_atom_num, inverse_atom_num):
        df = pd.read_csv(label_path)
        self.total_atom_num = total_atom_num
        self.inverse_atom_num = inverse_atom_num
        self.label = np.array(df['label'])
        self.x = df['X'][0:total_atom_num-inverse_atom_num]
        self.y = df['Y'][0:total_atom_num-inverse_atom_num]
        self.z = df['Z'][0:total_atom_num-inverse_atom_num]
        self.a_num = df['atomnum'][0:total_atom_num]
        self.atom =df['atom'][0:total_atom_num]

    def textreader(self,result_path):
        f = open(result_path)
        line = f.read()
        return line
    
    def appendwriter(self, text_path,write_data):
        f = open(text_path, 'a')
        f.write(write_data)
        f.close()
    
    def newwriter(self, text_path,write_data):
        f = open(text_path, 'w')
        f.writelines(write_data)
        f.close()


    def writer(self, input):
        input1 = np.reshape(input, (-1,3))
        atom = self.atom
        atom = np.array(atom)
        # atom = atom[~np.isnan(atom)]
        atom = np.reshape(atom, (self.total_atom_num,1))
        x = np.array(self.x)
        y = np.array(self.y)
        z = np.array(self.z)
        n_line = np.full((self.total_atom_num,1), '\n')
        x = np.reshape(x, (-1,1))
        y = np.reshape(y, (-1,1))
        z = np.reshape(z, (-1,1))
        o = np.concatenate([x, y], 1)
        o = np.concatenate([o, z], 1)
        o = np.concatenate([o, input1], 0)
        o = np.concatenate([atom, o], 1)
        o = np.concatenate([o, n_line],1)
        for i in range(len(o)):
            for j in range(len(o[i])):
                if type(o[i][j])==str:
                    pass
                else:     
                    o[i][j] = str(o[i][j])
        o = np.reshape(o, (self.total_atom_num*5,))
        o = o.tolist()
        o.insert(-1, "\n")
        o = ' '.join(o)
        template = self.textreader('package/template.txt')
        template = template.split()
        template.insert(1, "\n")
        template.insert(3, "\n")
        template.insert(5, "\n")
        template.insert(9, "\n")
        template.insert(9, "\n")
        template.insert(14, "\n")
        template.insert(14, "\n")
        template.insert(19, "\n")
        template.insert(7, " ")
        template.insert(9, " ")
        template.insert(19, " ")

        self.newwriter('input.txt',template)
        self.appendwriter('input.txt',o)

        os.rename('input.txt', 'input.gjf')



    def uvvis(self, wavelength, oscillatorstrength, wavelength_x):
        spe0 = 1.3062974*10**8*oscillatorstrength[0]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[0]))*(1/967)*10**7)**2)
        spe1 = 1.3062974*10**8*oscillatorstrength[1]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[1]))*(1/967)*10**7)**2)
        spe2 = 1.3062974*10**8*oscillatorstrength[2]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[2]))*(1/967)*10**7)**2)
        spe3 = 1.3062974*10**8*oscillatorstrength[3]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[3]))*(1/967)*10**7)**2)
        spe4 = 1.3062974*10**8*oscillatorstrength[4]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[4]))*(1/967)*10**7)**2)
        spe5 = 1.3062974*10**8*oscillatorstrength[5]*(1/967)*math.exp(-(((1/wavelength_x)-(1/wavelength[5]))*(1/967)*10**7)**2)
        spe =spe0+spe1+spe2+spe3+spe4+spe5
        return spe

    def find(self, file):
        index1 = []
        index2 = []
        index3 = []
        index4 = []
        index5 = []
        index6 = []
        index129 = []
        index_small = []
        file1 =file.split()
        for i in range(len(file1)):
            if file1[i] =='129':
                index129.append(i)
            else:
                pass
        for i in range(len(file1)):
            if file1[i] =='Small':
                index_small.append(i)
            else:
                pass
        if "129" in file1 and file1[index129[-1] + 1]=='cycles':
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        elif "Small" in file1 and file1[index_small[-1] + 1]=='interatomic':
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        else:
            for i in range(len(file1)):
                if file1[i] =='1:':
                    index1.append(i)
                else:
                    pass
            for i in range(len(file1)):
                if file1[i] =='2:':
                    index2.append(i)
                else:
                    pass
            for i in range(len(file1)):
                if file1[i] =='3:':
                    index3.append(i)
                else:
                    pass
            for i in range(len(file1)):
                if file1[i] =='4:':
                    index4.append(i)
                else:
                    pass
            for i in range(len(file1)):
                if file1[i] =='5:':
                    index5.append(i)
                else:
                    pass
            for i in range(len(file1)):
                if file1[i] =='6:':
                    index6.append(i)
                else:
                    pass
            ramda1 = float(file1[index1[-1] + 4])
            os1 = file1[index1[-1] + 6]
            os1 = float(os1.replace('f=', ''))
            ramda2 = float(file1[index2[-1] + 4])
            os2 = file1[index2[-1] + 6]
            os2 = float(os2.replace('f=', ''))
            ramda3 = float(file1[index3[-1] + 4])
            os3 = file1[index3[-1] + 6]
            os3 = float(os3.replace('f=', ''))
            ramda4 = float(file1[index4[-1] + 4])
            os4 = file1[index4[-1] + 6]
            os4 = float(os4.replace('f=', ''))
            ramda5 = float(file1[index5[-1] + 4])
            os5 = file1[index5[-1] + 6]
            os5 = float(os5.replace('f=', ''))
            ramda6 = float(file1[index6[-1] + 4])
            os6 = file1[index6[-1] + 6]
            os6 = float(os6.replace('f=', ''))
            return ramda1, os1, ramda2, os2, ramda3, os3, ramda4, os4, ramda5, os5, ramda6, os6, True


    def reader(self, input):
        os.rename(input, 'output.txt')
        wl = pd.read_csv('package/wavelength.csv')
        wl = np.array(wl)
        wl = np.reshape(wl, (500,))
        wl = np.delete(wl, 0)
        result_f = self.textreader('output.txt')
        rm1, os1, rm2, os2, rm3, os3, rm4, os4, rm5, os5, rm6, os6, judge = self.find(result_f)
        if judge:
            ramda_list = [rm1, rm2, rm3, rm4, rm5, rm6]
            osi_list = [os1, os2, os3, os4, os5, os6]
            intensity_list = []
            for i in range(len(wl)):
                intensity_list.append(self.uvvis(ramda_list, osi_list, wl[i]))
            intensity_list.insert(0, 0)
            intensity_list = np.reshape(np.array(intensity_list),(1,500))
            return intensity_list, True
        else:
            intensity_list = np.zeros((1, 499))
            intensity_list = np.append(intensity_list, 1)
            intensity_list = np.reshape(intensity_list,(1,500))
            return intensity_list, False

    
    def step(self, state):
        print(np.array(state))
        self.writer(state)
        gaussian = Popen('g16 input.gjf',shell=True )
        gaussian.wait()
        intens,done1 = self.reader('input.log')
        intens = np.array(intens)
        peak_target = np.argmax(self.label) * 1.6
        peak_predict = np.argmax(intens) * 1.6
        loss = float(abs(peak_target - peak_predict) / 400)
        return intens
