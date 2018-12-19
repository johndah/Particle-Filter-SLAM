from numpy import *


def motion_model_prediction(v, omega, S, R, delta_t):
    # Calculates prediction on the particle set S given v, omega, S, R and delta_t
    M = size(S[1,:]) # Number of particles
    dx = array([v*delta_t*cos(S[2,:])])
    dy = array([v*delta_t*sin(S[2,:])-1e-3])  # simulating error in motion model
    dtheta = ones([1, M])*(omega*delta_t+1e-2)  # simulating error in motion
    u = concatenate((dx, dy, dtheta))
    noise = dot(R,random.randn(3, M))  # Element wise multiplication
    print(u.shape)
    print(noise.shape)
    S[0:3,:] = S[:3,:] + u + noise
    return S


def motion_model_gt(v, omega, mu, delta_t):
    # Calculates ground truth u given v, omega, mu, and delta_t; u = (dx,dy,dtheta)
    dx = v*delta_t*cos(mu[2])
    dy = v*delta_t*sin(mu[2])
    dtheta = omega*delta_t
    u = [dx, dy, dtheta]
    return u

'''
def motion_path():
    start = 0
    stop = 2
    for t in linspace(start,stop,(stop-start)/delta_t):
'''

