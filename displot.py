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

    rl_cx = []
    rl_cy = []

    pp_cx = []
    pp_cy = []

    stanely_cx = []
    stanely_cy = []
    #read file line by line
    while True:

        x.append(mpc.state.x)
        y.append(mpc.state.y)
        yaw.append(mpc.state.yaw)
        v.append(mpc.state.v)
        t.append(time)
        d.append(di)
        a.append(ai)
    
        if mpc.show_animation:
            plt.cla()
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            mpc.plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2)) +
                      ", yaw:" + str(round(state.yaw, 2)))
            plt.pause(0.0001)
    
    return t, x, y, yaw, v, d, a

if __name__ == '__main__':
    main()
