from numpy import *


def motion_model_prediction(v, omega, mu, delta_t):
    # Calculates prediction u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v*delta_t*cos(mu[2])
    dy = v*delta_t*sin(mu[2])-1e-3  # simulating error in motion model
    dtheta = omega*delta_t+1e-2  # simulating error in motion model
    u = [dx, dy, dtheta]
    return u


def motion_model_gt(v, omega, mu, delta_t):
    # Calculates ground truth u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v*delta_t*cos(mu[2])
    dy = v*delta_t*sin(mu[2])
    dtheta = omega*delta_t
    u = [dx, dy, dtheta]
    return u

def motion_model(v, omega, robot_poses, delta_t, i):
    # Calculates ground truth u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v*delta_t*cos(robot_poses[2, i])
    dy = v*delta_t*sin(robot_poses[2, i])
    dtheta = omega*delta_t
    robot_poses[0, i+1] = robot_poses[0, i] + dx
    robot_poses[1, i+1] = robot_poses[1, i] + dy
    robot_poses[2, i+1] = robot_poses[2, i] + dtheta

    return robot_poses




