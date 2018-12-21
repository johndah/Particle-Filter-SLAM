from numpy import *
import matplotlib.pyplot as plt
from motion import *


def particle_init(window, M, start_pose = []):
    # initializes the particle set of M particles inside given window
    # window should be  [x_min, x_max, y_min, y_max]
    sigma_xy = 1e-1  # Variance in starting position for known pose for x and y
    sigma_theta = 1e-1  # Variance in starting position for known pose for theta
    S = zeros([4, M])
    if not start_pose:  # If start_pose is empty, wont be used
        S[0, :] = random.uniform(window[0], window[1], [1, M])
        S[1, :] = random.uniform(window[2], window[3], [1, M])
        S[2, :] = random.uniform(-pi/2, pi/2, [1, M])
    else:
        S = zeros([4, M])  # If start_pose is given the particle set will be gaussians around the starting position
        S[0, :] = start_pose[0]+random.randn(1, M)*sigma_xy
        S[1, :] = start_pose[1]+random.randn(1, M)*sigma_xy
        S[1, :] = start_pose[2]+random.randn(1, M)*sigma_theta
    return S


def plot_particle_set(S):
    #  Plots particle set S in figure figure
    #  S has dimensions 4xM where M in the number of particles
    plt.scatter(S[0,:],S[1,:])


def systematic_resample(S):  # Must include map resample
    M = S.shape[1]
    cdf = cumsum(S[3,:])

    rand = random.uniform(0,1/M,1)
    print('rand')
    print(rand)
    S_new = zeros(S.shape)
    for i in arange(0,M,1):
        print(cdf >= rand + (i ) / M)
        c = argmax(cdf >= rand+(i)/M)
        print('c')
        print(c)
        S_new[:,i] = S[:,c]

    return S_new


def weight(S, Psi, outlier):
    # Adds weights to the last row in S
    # Psi in on the [n, M] where n is the number of measurements an M is the number of particles
    # outlier contains information about measurement outliers
    pz = prod(Psi[where(1-outlier)], 1)  # Non normalized weights without outliers
    w = pz*1/sum(pz)  # Normalization
    S[4,:] = w
    return S

def measurement_model(W, S):
    # W is the location of the landmarks on each particles map. Shape [2*landmarks, particles]
    # S is the particle set. Shape [4, particles]
    # h is predicted measurements. Shape [2*landmarks, particles]
    no_landmarks = W.shape[0]/2
    xindices = arange(0,no_landmarks+1,2)
    yindices = xindices + 1

    h = [sqrt(square(W[xindices] - S[0,:]) + square(W[yindices] - S[1,:]) ), # inputa varannan
    atan2(W(yindices) - S(2,:), W(xindices) - S(1,:)) - S(3,:)];

    h(2,:) = mod(h(2,:)+pi, 2 * pi)-pi;

    return h
'''
def main():
    window = [0,5,0,5]
    S = particle_init(window, 100)
    fig1 = plt.figure()
    plot_particle_set(S, fig1)

    R = diag([0,0,0])
    delta_t = 1
    S = motion_model_prediction(1, 0, S, R, delta_t)
    fig2 = plt.figure()
    plot_particle_set(S, fig2)
    plt.show(fig1)
    '''
def main():
    S = array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[0,1/6,2/6,3/6]])
    # print(S)
    S = systematic_resample(S)
    print(S)

main()

