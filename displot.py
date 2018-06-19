"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys

import numpy as np
import math
import matplotlib.pyplot as plt

#dependent function
def main():
    print(__file__ + " start!!")

    #load map
    map_cx = []
    map_cy = []
    with open('./log/all.road') as mapfile:
        line = mapfile.readline()
        while line:
            args = line.split('\t')
            map_cx.append(float(args[0]))
            map_cy.append(float(args[1]))
            line = mapfile.readline()

    rlfile = open('./log/rl.track')
    rl_cx = []
    rl_cy = []
    rlline = rlfile.readline()

    ppfile = open('./log/pp.track')
    pp_cx = []
    pp_cy = []
    ppline = ppfile.readline()

    stanelyfile = open('./log/stanely.track')
    stanely_cx = []
    stanely_cy = []
    stanelyline = stanelyfile.readline()

    #read file line by line
    while rlline and ppline and stanelyline:

        rlargs = rlline.split('\t')
        rl_cx.append(float(rlargs[0]))
        rl_cy.append(float(rlargs[1]))
        sumrl = 12
        while sumrl > 0:
            rlline = rlfile.readline()
            sumrl -= 1

        ppargs = ppline.split('\t')
        pp_cx.append(float(ppargs[0]))
        pp_cy.append(float(ppargs[1]))
        sumpp = 5
        while sumpp > 0:
            ppline = ppfile.readline()
            sumpp -= 1

        stanelyargs = stanelyline.split('\t')
        stanely_cx.append(float(stanelyargs[0]))
        stanely_cy.append(float(stanelyargs[1]))
        sumstanely = 5
        while sumstanely > 0:
            stanelyline = stanelyfile.readline()
            sumstanely -= 1

        plt.plot(map_cx, map_cy, "-r", label="course", linewidth=1)
        plt.plot(rl_cx, rl_cy, "-b", label="trajectory", linewidth=1)
        plt.plot(pp_cx, pp_cy, "-k", label="trajectory", linewidth=1)
        plt.plot(stanely_cx, stanely_cy, "-m", label="trajectory", linewidth = 1)
        plt.grid(True)
        plt.pause(0.001)
    
        plt.plot(map_cx, map_cy, "-r", label="course", linewidth=1)
        plt.plot(rl_cx, rl_cy, "-b", label="trajectory", linewidth=1)
        plt.plot(pp_cx, pp_cy, "-k", label="trajectory", linewidth=1)
        plt.plot(stanely_cx, stanely_cy, "-m", label="trajectory", linewidth = 1)
        plt.grid(True)

if __name__ == '__main__':
    main()
