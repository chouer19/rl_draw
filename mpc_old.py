"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
sys.path.append("./libs/CubicSpline/")

import numpy as np
import math
import cvxpy
import matplotlib.pyplot as plt
import cubic_spline_planner

#dependent function
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

class Road:
    def __int__(self):
        self.cx = [0]
        self.cy = [0]
        self.cyaw = [0]
        self.ck = [0]
        self.sp = [0]
        self.dl = 1

class Vehicle:

    def __init__(self):
        self.LENGTH = 4.5  # [m]
        self.WIDTH = 2.0  # [m]
        self.BACKTOWHEEL = 1.0  # [m]
        self.WHEEL_LEN = 0.3  # [m]
        self.WHEEL_WIDTH = 0.2  # [m]
        self.TREAD = 0.7  # [m]
        self.WB = 2.5  # [m]

        self.MAX_STEER = math.radians(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = math.radians(30.0)  # maximum steering speed [rad/s]
        self.MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
        self.MIN_SPEED = 0.0 / 3.6  # minimum speed [m/s]
        self.MAX_ACCEL = 1.0  # maximum accel [m/ss]
        self.MAX_BRAKE = 1.0  # maximum brake [m/ss]

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.z = 0
        self.v = v
        self.steer = 0
        self.yaw = yaw
        self.pitch = 0
        self.roll = 0
        self.predelta = None

class Mpc:
    def __init__(self):
        self.NX = 4  # x = x, y, v, yaw
        self.NU = 2  # a = [accel, steer]
        self.T = 5  # horizon length
        self.HorizonLength = 5
        self.HorizonLength = 10

        self.R = np.diag([0.01, 0.01])  # input cost matrix
        self.Rd = np.diag([0.01, 1.0])  # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.GOAL_DIS = 1.5  # goal distance
        self.GOAL_DIS = 5  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 500.0  # max simulation time

        # iterative paramter
        self.MAX_ITER = 5  # Max iteration
        self.MAX_ITER = 8  # Max iteration
        self.DU_TH = 0.1  # iteration finish param
        self.N_IND_SEARCH = 10  # Search index number
    
        self.show_animation = True

        self.DT = 0.2 # time tick (s)

        self.vehicle = Vehicle()
        self.road = Road()
        self.state = State()


    def get_linear_model_matrix(self,v, phi, delta):
    
        A = np.matrix(np.zeros((self.NX, self.NX)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta)
    
        B = np.matrix(np.zeros((self.NX, self.NU)))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.vehicle.WB * math.cos(delta) ** 2)
    
        C = np.matrix(np.zeros((self.NX, 1)))
        C[0, 0] = self.DT * v * math.sin(phi) * phi
        C[1, 0] = - self.DT * v * math.cos(phi) * phi
        C[3, 0] = v * delta / (self.vehicle.WB * math.cos(delta) ** 2)
    
        return A, B, C
    
    
    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
        outline = np.matrix([[-self.vehicle.BACKTOWHEEL, (self.vehicle.LENGTH - self.vehicle.BACKTOWHEEL), (self.vehicle.LENGTH - self.vehicle.BACKTOWHEEL), -self.vehicle.BACKTOWHEEL, -self.vehicle.BACKTOWHEEL],
                             [self.vehicle.WIDTH / 2, self.vehicle.WIDTH / 2, - self.vehicle.WIDTH / 2, -self.vehicle.WIDTH / 2, self.vehicle.WIDTH / 2]])
    
        fr_wheel = np.matrix([[self.vehicle.WHEEL_LEN, -self.vehicle.WHEEL_LEN, -self.vehicle.WHEEL_LEN, self.vehicle.WHEEL_LEN, self.vehicle.WHEEL_LEN],
                              [-self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, -self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, -self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD]])
    
        rr_wheel = np.copy(fr_wheel)
    
        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1
    
        Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                          [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
                          [-math.sin(steer), math.cos(steer)]])
    
        fr_wheel = (fr_wheel.T * Rot2).T
        fl_wheel = (fl_wheel.T * Rot2).T
        fr_wheel[0, :] += self.vehicle.WB
        fl_wheel[0, :] += self.vehicle.WB
    
        fr_wheel = (fr_wheel.T * Rot1).T
        fl_wheel = (fl_wheel.T * Rot1).T
    
        outline = (outline.T * Rot1).T
        rr_wheel = (rr_wheel.T * Rot1).T
        rl_wheel = (rl_wheel.T * Rot1).T
    
        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y
    
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
                 np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
                 np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
                 np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
                 np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(x, y, "*")
    
    
    def update_state(self,state, a, delta):
    
        # input check
        if delta >= self.vehicle.MAX_STEER:
            delta = self.vehicle.MAX_STEER
        elif delta <= -self.vehicle.MAX_STEER:
            delta = -self.vehicle.MAX_STEER
    
        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.vehicle.WB * math.tan(delta) * self.DT
        state.v = state.v + a * self.DT
    
        if state. v > self.vehicle.MAX_SPEED:
            state.v = self.vehicle.MAX_SPEED
        elif state. v < self.vehicle.MIN_SPEED:
            state.v = self.vehicle.MIN_SPEED
    
        return state
    
    
    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        """
    
        if oa is None or od is None:
            #oa = [0.0] * T
            #od = [0.0] * T
            oa = [0.0] * self.HorizonLength
            od = [0.0] * self.HorizonLength
    
        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break
        else:
            print("Iterative is max iter")
    
        return oa, od, ox, oy, oyaw, ov
    
    def linear_mpc_control(self,xref, xbar, x0, dref):
        """
        linear mpc control
    
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
    
        x = cvxpy.Variable(self.NX, self.HorizonLength + 1)
        u = cvxpy.Variable(self.NU, self.HorizonLength)
    
        cost = 0.0
        constraints = []
    
        for t in range(self.HorizonLength):
            cost += cvxpy.quad_form(u[:, t], self.R)
    
            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)
    
            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]
    
            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t])
                                < self.vehicle.MAX_DSTEER * self.DT]
    
        #cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
        cost += cvxpy.quad_form(xref[:, self.HorizonLength] - x[:, self.HorizonLength], self.Qf)
    
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.vehicle.MAX_SPEED]
        constraints += [x[2, :] >= self.vehicle.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) < self.vehicle.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) < self.vehicle.MAX_STEER]
    
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(verbose=False)
    
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
    
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
    
        return oa, odelta, ox, oy, oyaw, ov
    
    
    def calc_ref_trajectory(self):
        #xref = np.zeros((NX, T + 1))
        xref = np.zeros((self.NX, self.HorizonLength + 1))
        #dref = np.zeros((1, T + 1))
        dref = np.zeros((1, self.HorizonLength + 1))
        ncourse = len(self.road.cx)
    
        ind, _ = calc_nearest_index(self.state, self.road.cx, self.road.cy, self.road.cyaw)
    
        xref[0, 0] = self.road.cx[ind]
        xref[1, 0] = self.road.cy[ind]
        xref[2, 0] = self.road.sp[ind]
        xref[3, 0] = self.road.cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0
    
        travel = 0.0
    
        #for i in range(T + 1):
        for i in range(self.HorizonLength + 1):
            travel += abs(self.state.v) * self.DT
            dind = int(round(travel / self.road.dl))
    
            if (ind + dind) < ncourse:
                xref[0, i] = self.road.cx[ind + dind]
                xref[1, i] = self.road.cy[ind + dind]
                xref[2, i] = self.road.sp[ind + dind]
                xref[3, i] = self.road.cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = self.road.cx[ncourse - 1]
                xref[1, i] = self.road.cy[ncourse - 1]
                xref[2, i] = self.road.sp[ncourse - 1]
                xref[3, i] = self.road.cyaw[ncourse - 1]
                dref[0, i] = 0.0
    
        return xref, ind, dref
    
    #dependent function
    def predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
    
        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.HorizonLength + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw
    
        return xbar

    def do_simulation(self,cx, cy, cyaw, ck, sp, dl):
        """
        Simulation
    
        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]
    
        """
    
        goal = [cx[-1], cy[-1]]
        state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    
        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]
        target_ind, _ = calc_nearest_index(state, cx, cy, cyaw)
    
        odelta, oa = None, None
    
        cyaw = smooth_yaw(cyaw)
    
        while self.MAX_TIME >= time:
            xref, target_ind, dref = self.calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, target_ind)
    
            x0 = [state.x, state.y, state.v, state.yaw]  # current state
    
            #kenerl
            oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
                xref, x0, dref, oa, odelta)
    
            if odelta is not None:
                di, ai = odelta[0], oa[0]
    
            state = self.update_state(state, ai, di)
            time = time + self.DT
    
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)
            d.append(di)
            a.append(ai)
    
            if check_goal(state, goal, target_ind, len(cx)):
                print("Goal")
                break
    
            if self.show_animation:
                plt.cla()
                if ox is not None:
                    plt.plot(ox, oy, "xr", label="MPC")
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(x, y, "ob", label="trajectory")
                plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                self.plot_car(state.x, state.y, state.yaw, steer=di)
                plt.axis("equal")
                plt.grid(True)
                plt.title("Time[s]:" + str(round(time, 2)) +
                          ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
                plt.pause(0.0001)
    
        return t, x, y, yaw, v, d, a

#dependet function
def calc_nearest_index(state, cx, cy, cyaw):

    N_IND_SEARCH = 10
    #dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    #dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    dx = [state.x - icx for icx in cx[:]]
    dy = [state.y - icy for icy in cy[:]]

    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]

    min_d = min(d)

    #ind = d.index(min_d) + pind
    ind = d.index(min_d)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        min_d *= -1

    return ind, min_d

    
#dependent function
def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

#dependent function
def calc_speed_profile2(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            speed_profile[i] = target_speed * abs(math.cos(dangle))

    speed_profile[-1] = 0.0
    speed_profile[-2] = 0.0

    return speed_profile


#dependent function
def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

#dependent function
def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.sqrt(dx ** 2 + dy ** 2)

    #if (d <= GOAL_DIS):
    if (d <= 5):
        isgoal = True
    else:
        isgoal = False

    if abs(tind - nind) >= 5:
        isgoal = False

    #if (abs(state.v) <= STOP_SPEED):
    if (abs(state.v) <= 5):
        isstop = True
    else:
        isstop = False

    if isgoal and isstop:
        return True

    return False

#dependent function
def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

#dependent function
def get_switch_back_course(dl):
    ax = [0.0, 10.0, 20.0, 30.0, 35.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 35.0, 35.0, 30.0]
    ay = [20.0, 30.0, 35.0, 30.0]
    ax = [35.0, 35.0, 35.0, 33.0, 30.0, 27.0, 23.0, 20.0, 15.0, 10, 11, 14, 16, 12.0, 10.0, 8.0, 6.0, 3.5, 0.0]
    ay = [00.0, 10.0, 25.0, 35.0, 34.0, 32.0, 30.0, 28.0, 29.0, 25.0, 24.0, 20.0, 15.0, 13.0, 10.0, 6.0, 4.0, 0.0 ]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck

#dependent function
def main():
    print(__file__ + " start!!")

    #get waypoints of road,
    #no need to spline in rl experiments
    dl = 1.0  # course tick
    dl = 0.2  # course tick
    #  cx, cy, cyaw, ck = get_forward_course(dl)
    # get a spline road
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    # calc speed
    #sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    sp = calc_speed_profile2(cx, cy, cyaw, 30 / 3.6)

    #ax = [0.0, 10.0, 20.0, 30.0, 35.0]
    #sp = [1, 10, 20, 30, 15, 15, 15, 20, 20, 15, 20, 25, 30, 35, 20, 15, 10, 20, 25, 20, 16, 0]
    #ax = [35.0, 35.0, 35.0, 30.0, 25.0, 20.0, 15.0, 12.0, 5.0, 10, 11, 14, 16, 12.0, 10.0, 8.0, 6.0, 3.0, 0.0]
    mpc = Mpc()
    mpc.road.cx = cx
    mpc.road.cy = cy
    mpc.road.cyaw = cyaw
    mpc.road.ck = ck
    mpc.road.sp = sp
    mpc.road.dl = dl
    #t, x, y, yaw, v, d, a = mpc.do_simulation(cx, cy, cyaw, ck, sp, dl)
    def do_simulation2(mpc):
        #get state
        #state = State(x=mpc.cx[0], y=mpc.cy[0], yaw=mpc.cyaw[0], v=0.0)
        mpc.state = State(x=mpc.road.cx[0], y=mpc.road.cy[0], yaw=mpc.road.cyaw[0], v=0.0)
        state = mpc.state
    
        time = 0.0
        x = [mpc.state.x]
        y = [mpc.state.y]
        yaw = [mpc.state.yaw]
        v = [mpc.state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]
        # get index of nearest point on road
        target_ind, _ = calc_nearest_index(mpc.state, mpc.road.cx, mpc.road.cy, mpc.road.cyaw)
    
        odelta, oa = None, None
    
        # convert illegal yaw to be legal
        mpc.road.cyaw = smooth_yaw(mpc.road.cyaw)
    
        while 16 >= time:
        #while True:

            #xref, target_ind, dref = mpc.calc_ref_trajectory(
            #    state, cx, cy, cyaw, ck, sp, dl, target_ind)
            xref, target_ind, dref = mpc.calc_ref_trajectory()
    
            x0 = [mpc.state.x, mpc.state.y, mpc.state.v, mpc.state.yaw]  # current state
    
            #kenerl
            oa, odelta, ox, oy, oyaw, ov = mpc.iterative_linear_mpc_control(
                xref, x0, dref, oa, odelta)
    
            if odelta is not None:
                di, ai = odelta[0], oa[0]
    
            # get new state
            mpc.state = mpc.update_state(mpc.state, ai, di)
            time = time + mpc.DT
    
            x.append(mpc.state.x)
            y.append(mpc.state.y)
            yaw.append(mpc.state.yaw)
            v.append(mpc.state.v)
            t.append(time)
            d.append(di)
            a.append(ai)
    
            if mpc.show_animation:
                plt.cla()
                plt.plot(cx, cy, "-r", label="course")
                #plt.plot(x, y, "ob", label="trajectory")
                plt.plot(x, y, "-b", label="trajectory")
                mpc.plot_car(state.x, state.y, state.yaw, steer=di)
                plt.axis("equal")
                plt.grid(True)
                plt.title("Time[s]:" + str(round(time, 2)) +
                          ", yaw:" + str(round(state.yaw, 2)))
                plt.pause(0.0001)
    
        return t, x, y, yaw, v, d, a
    t, x, y, yaw, v, d, a = do_simulation2(mpc)

    if mpc.show_animation:
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main()
