import sys
sys.path.append("../package")
import numpy as np
import random
from environment_PSO import Env
import csv
import pandas as pd
import time

T = 20  #Number of loops
position_min, position_max = -2, 2
total_atom_num = 12
inverse_atom_num = 1   #number of inverse designed atom
model_name = 'Cd6Se6/model_Cd6Se6.pkl'    #spectra predict model
target_data = 'Cd6Se6/Cd6Se6_abs.csv'     #target structure and spectra data
env = Env(model_name, target_data, total_atom_num, inverse_atom_num)

#evaluation positino of each particle
def criterion(particle):
    r = env.step(particle)
    return r

#Update the position of the particle
def update_position(_position, _velocity):
    new_position = np.array(_position) + np.array(_velocity)
    return new_position

#Update the velocity of the particles
def update_velocity(_ps, _vs, p, g, w=0.5, ro_max=0.2):
    ro1 = random.uniform(0, ro_max)
    ro2 = random.uniform(0, ro_max)
    new_v = w * np.array(_vs) + ro1 * (np.array(p) - np.array(_ps)) + ro2 * (np.array(g) - np.array(_ps))
    return new_v


def main():
    N = 300   #number of particle
    df = pd.read_csv(target_data)
    target_position = np.array(df[['X','Y','Z']][total_atom_num-inverse_atom_num:total_atom_num]).reshape((1,-1))


    ps = [[target_position[0][i]+random.uniform(position_min, position_max) for i in range(3*inverse_atom_num)] for i in range(N)]
    vs = [[0 for i in range(3*inverse_atom_num)] for i in range(N)]
    personal_best_positions = list(ps)
    personal_best_scores = [criterion(p) for p in ps]
    best_particle = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[best_particle]
    start = time.time()
    
    for t in range(T):
        for n in range(N):
            p = personal_best_positions[n]
            #Update the position of the particle
            new_position = update_position(ps[n],vs[n])
            ps[n] = new_position
            #Update the velocity of the particles
            new_velocity = update_velocity(ps[n], vs[n], p, global_best_position)
            vs[n] = new_velocity
            #evaluation positino of each particle and update best position of each particle
            score = criterion(new_position)
            buffer = list(new_position) + [score]
            if score < personal_best_scores[n]:
                personal_best_scores[n] = score
                personal_best_positions[n] = new_position
        #Update the particle of best score
        best_particle = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_particle]

    print(time.time() - start)
    print('target position', target_position)
    print('optimized position', global_best_position)
    # print(min(personal_best_scores))

if __name__ == '__main__':
    main()
