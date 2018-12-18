from numpy import *


def particle_init(window, M, start_pose = []):
    # initializes the particle set of M particles inside given window
    # window should be  [x_min, x_max, y_min, y_max]
    sigma_xy = 1e-1  # Variance in starting position for known pose for x and y
    sigma_theta = 1e-1  # Variance in starting position for known pose for theta
    S = zeros([4, M])
    if not start_posea:  # If start_pose is empty
        S[0, :] = random.uniform(window[0], window[1], [1, M])
        S[1, :] = random.uniform(window[0], window[1], [1, M])
        S[1, :] = random.uniform(-pi/2, pi/2, [1, M])
    else:
        S = zeros([4, M])  # If start_pose is given the particle set will be gaussians around the starting position
        S[0, :] = start_pose[0]+random.randn(1, M)*sigma_xy
        S[1, :] = start_pose[1]+random.randn(1, M)*sigma_xy
        S[1, :] = start_pose[2]+random.randn(-pi / 2, pi / 2, [1, M])*sigma_theta
    return S
