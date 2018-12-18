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




