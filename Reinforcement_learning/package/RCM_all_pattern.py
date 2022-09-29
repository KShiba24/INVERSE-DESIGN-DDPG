import numpy as np
import pandas as pd
import math
import itertools
import csv


class Coulomb_matrix_all():
    def make_input_data(self, input,a_num, x,y,z):
        input = np.reshape(input, (-1,3))
        a_num = np.array(a_num)
        a_num = a_num[~np.isnan(a_num)]
        a_num = np.reshape(a_num, (-1,1))
        len_atom = len(a_num)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x = np.reshape(x, (-1,1))
        y = np.reshape(y, (-1,1))
        z = np.reshape(z, (-1,1))
        o = np.concatenate([x, y, z], 1)
        o = np.concatenate([o, input], 0)
        o = np.concatenate([a_num, o], 1)
        x = o[:len_atom, 0]
        y = o[:len_atom, 1]
        z = o[:len_atom, 2]
        structure = pd.DataFrame(o,columns=['atomnum', 'X', 'Y', 'Z'])
        return structure
        
    def make_matrix(self,input):
        a_num = input['atomnum']
        x = input['X']
        y = input['Y']
        z = input['Z']
        a_num = np.array(a_num)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        r = [0] * (len(x))
        for n in range( len(x) ):
            r[n] = [0] * (len(x))
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    r[i][j] = 0.5 *( a_num[i]**2.4)
                else:
                    r[i][j] = (a_num[i] * a_num[j])/(math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2))
        train = np.reshape(r, (1,-1))
        return train


    #inputは各原子ごとの行をまとめていれる。出力はその原子の行の組み合わせ
    def all_pattern_sort_index(self, input):
        # print(input)
        length = len(input)
        index_list = []
        for i in range(length):
            index_list.append(i)
        comb_list = []
        for comb_num in itertools.permutations(index_list, length):
            comb_list.append(list(comb_num))
        return comb_list


    def main(self, data_path, f_name, cm_save_name, label_save_name):      #読み込むcsvの通し番号
        df = pd.read_csv(f_name,index_col=0)
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
            se_comb_list = self.all_pattern_sort_index(se_row)
            cd_comb_list = self.all_pattern_sort_index(cd_row)
            te_comb_list = self.all_pattern_sort_index(te_row)


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
                        #if i == 0 and k == 0 and m == 0:
                          #  print(input_position)
                        matrix = self.make_matrix(input_position)
                        with open(cm_save_name, 'a') as f: 
                            writer = csv.writer(f)
                            writer.writerow(matrix[0])
                        with open(label_save_name, 'a') as f: 
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
            se_comb_list = self.all_pattern_sort_index(se_row)
            cd_comb_list = self.all_pattern_sort_index(cd_row)
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
                    #if i == 0 and k == 0:
                       # print(input_position)
                    matrix = self.make_matrix(input_position)
                    with open(cm_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(matrix[0])
                    with open(label_save_name, 'a') as f:  
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
            cd_comb_list = self.all_pattern_sort_index(cd_row)
            te_comb_list = self.all_pattern_sort_index(te_row)
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
                    #if i == 0 and m == 0:
                     #   print(input_position)
                    matrix = self.make_matrix(input_position)
                    with open(cm_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(matrix[0])
                    with open(label_save_name, 'a') as f:   
                        writer = csv.writer(f)
                        writer.writerow(target)

