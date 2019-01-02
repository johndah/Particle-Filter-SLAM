from numpy import *


def motion_model_prediction(S, v, omega, R, delta_t):
    # Calculates prediction on the particle set S given v, omega, S, R and delta_t
    M = size(S, 1)  # Number of particles
    dx = array([v * delta_t * cos(S[2, :])])
    dy = array([v * delta_t * sin(S[2, :]) - 0*1e-3])  # simulating error in motion model
    dtheta = ones([1, M]) * (omega * delta_t + 0*1e-2)  # simulating error in motion
    dy = array([v * delta_t * sin(S[2, :]) - 1e-3])  # simulating error in motion model
    dtheta = ones([1, M]) * (omega * delta_t + 1e-4)  # simulating error in motion
    u = concatenate((dx, dy, dtheta))
    noise = dot(R, random.randn(3, M))  # Noise with covariance R
    S[:3, :] = S[:3, :] + u + noise

    return S


def motion_model_gt(v, omega, mu, delta_t):
    # Calculates ground truth u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v * delta_t * cos(mu[2])
    dy = v * delta_t * sin(mu[2])
    dtheta = omega * delta_t
    u = [dx, dy, dtheta]
    return u


def motion_model(v, omega, robot_poses, delta_t, i):
    # Calculates ground truth u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v * delta_t * cos(robot_poses[2, i])
    dy = v * delta_t * sin(robot_poses[2, i])
    dtheta = omega * delta_t
    robot_poses[0, i + 1] = robot_poses[0, i] + dx
    robot_poses[1, i + 1] = robot_poses[1, i] + dy
    robot_poses[2, i + 1] = robot_poses[2, i] + dtheta

    return robot_poses
