from re import S
import numpy as np
import pandas as pd
import math
import itertools
import csv
import matplotlib.pyplot as plt
from RCM_all_pattern import Coulomb_matrix_all


cm_all = Coulomb_matrix_all()
class Coulomb_matrix_part():
    #inputは各原子ごとの行をまとめていれる。出力はその原子の行の組み合わせ
    def all_pattern_sort_index(self, input):
        length = len(input)
        index_list = []
        for i in range(length):
            index_list.append(i)
        comb_list = []
        for comb_num in itertools.permutations(index_list, length):
            comb_list.append(list(comb_num))
        return comb_list

    def random_pattern_sort_index(self, input):
        length = len(input)
        comb_list = []
        l_list = []
        l = []
        for i in range(length):
            l.append(i)
        for j in range(24): #24
            l_list.append(l.copy())
        for m in range(24): #24
            index_list = []
            for n in range(length):
                inde_list = []
                r_n = np.random.randint(0,len(l_list[m]),(1,1))
                stack = l_list[m].pop(r_n[0][0])
                index_list.append(stack)
            comb_list.append(index_list.copy())
        return comb_list


    def main(self, data_path, f_name, cm_save_name, label_save_name):
        df = pd.read_csv(data_path+f_name,index_col=0)
        target = df['label']
        index = df.index
        if np.count_nonzero(index == 'Se')>0 and np.count_nonzero(index == 'Cd')>0 and np.count_nonzero(index == 'Te')>0:
            se_row = df.loc['Se']
            cd_row = df.loc['Cd']
            te_row = df.loc['Te']
            if type(se_row) == pd.core.series.Series:
                se_row = pd.DataFrame(se_row).T
            else:
                pass

            if type(cd_row) == pd.core.series.Series:
                cd_row = pd.DataFrame(cd_row).T
            else:
                pass

            if type(te_row) == pd.core.series.Series:
                te_row = pd.DataFrame(te_row).T
            else:
                pass
            if len(se_row) <=4:
                se_comb_list = self.all_pattern_sort_index(se_row)
            else:
                se_comb_list = self.random_pattern_sort_index(se_row)
            if len(cd_row) <= 4:
                cd_comb_list = self.all_pattern_sort_index(cd_row)
            else:
                cd_comb_list = self.random_pattern_sort_index(cd_row)
            if len(te_row) <= 4:
                te_comb_list = self.all_pattern_sort_index(te_row)
            else:
                te_comb_list = self.random_pattern_sort_index(te_row)

            for i in range(len(se_comb_list)):
                for j in range(len(se_comb_list[i])):
                    if j == 0:
                        se_list = pd.DataFrame(se_row.iloc[se_comb_list[i][j],:]).T
                    else:
                        se_list = pd.concat([se_list, pd.DataFrame(se_row.iloc[se_comb_list[i][j],:]).T])
                for k in range(len(cd_comb_list)):
                    for l in range(len(cd_comb_list[0])):
                        if l == 0:    
                            cd_list = pd.DataFrame(cd_row.iloc[cd_comb_list[k][l],:]).T
                        else:
                            cd_list = pd.concat([cd_list, pd.DataFrame(cd_row.iloc[cd_comb_list[k][l],:]).T])
                    for m in range(len(te_comb_list)):
                        for n in range(len(te_comb_list[0])):
                            if n == 0:
                                te_list = pd.DataFrame(te_row.iloc[te_comb_list[m][n],:]).T
                            else:
                                te_list = pd.concat([te_list, pd.DataFrame(te_row.iloc[te_comb_list[m][n],:]).T])
                        input_position = pd.concat([se_list, cd_list])
                        input_position = pd.concat([input_position, te_list])
                        matrix = cm_all.make_matrix(input_position)
                        with open(data_path+cm_save_name, 'a') as f: 
                            writer = csv.writer(f)
                            writer.writerow(matrix[0])
                        with open(data_path+label_save_name, 'a') as f: 
                            writer = csv.writer(f)
                            writer.writerow(target)



        elif np.count_nonzero(index == 'Se')>0 and np.count_nonzero(index == 'Cd')>0:
            se_row = df.loc['Se']
            cd_row = df.loc['Cd']
            if type(se_row) == pd.core.series.Series:
                se_row = se_row.T
            else:
                pass
            if type(cd_row) == pd.core.series.Series:
                cd_row = cd_row.T
            else:
                pass
            if len(se_row) <=4:
                se_comb_list = self.all_pattern_sort_index(se_row)
            else:
                se_comb_list = self.random_pattern_sort_index(se_row)
            if len(cd_row) <= 4:
                cd_comb_list = self.all_pattern_sort_index(cd_row)
            else:
                cd_comb_list = self.random_pattern_sort_index(cd_row)
            for i in range(len(se_comb_list)):
                for j in range(len(se_comb_list[0])):
                    if j == 0:
                        se_list = pd.DataFrame(se_row.iloc[se_comb_list[i][j],:]).T
                    else:
                        se_list = pd.concat([se_list, pd.DataFrame(se_row.iloc[se_comb_list[i][j],:]).T])
                for k in range(len(cd_comb_list)):
                    for l in range(len(cd_comb_list[0])):
                        if l == 0:
                            cd_list = pd.DataFrame(cd_row.iloc[cd_comb_list[k][l],:]).T
                        else:
                            cd_list = pd.concat([cd_list, pd.DataFrame(cd_row.iloc[cd_comb_list[k][l],:]).T])
                    input_position = pd.concat([se_list, cd_list])
                    matrix = cm_all.make_matrix(input_position)
                    with open(data_path+cm_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(matrix[0])
                    with open(data_path+label_save_name, 'a') as f:  
                        writer = csv.writer(f)
                        writer.writerow(target)



        elif np.count_nonzero(index == 'Cd')>0 and np.count_nonzero(index == 'Te')>0:
            cd_row = df.loc['Cd']
            te_row = df.loc['Te']
            if type(cd_row) == pd.core.series.Series:
                cd_row = cd_row.T
            else:
                pass
            if type(te_row) == pd.core.series.Series:
                te_row = te_row.T
            else:
                pass
            if len(cd_row) <= 4:
                cd_comb_list = self.all_pattern_sort_index(cd_row)
            else:
                cd_comb_list = self.random_pattern_sort_index(cd_row)
            if len(te_row) <= 4:
                te_comb_list = self.all_pattern_sort_index(te_row)
            else:
                te_comb_list = self.random_pattern_sort_index(te_row)
            for i in range(len(cd_comb_list)):
                for j in range(len(cd_comb_list[0])):
                    if j == 0:
                        cd_list = pd.DataFrame(cd_row.iloc[cd_comb_list[i][j],:]).T
                    else:
                        cd_list = pd.concat([cd_list, pd.DataFrame(cd_row.iloc[cd_comb_list[i][j],:]).T])
                for m in range(len(te_comb_list)):
                    for n in range(len(te_comb_list[0])):
                        if n == 0:
                            te_list = pd.DataFrame(te_row.iloc[te_comb_list[m][n],:]).T
                        else:
                            te_list = pd.concat([te_list, pd.DataFrame(te_row.iloc[te_comb_list[m][n],:]).T])
                    input_position = pd.concat([cd_list,te_list])
                    matrix = cm_all.make_matrix(input_position)
                    with open(data_path+cm_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(matrix[0])
                    with open(data_path+label_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(target)

