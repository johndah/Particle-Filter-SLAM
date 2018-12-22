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


def associate_known(S, measurements, W, lambda_Psi, Q, known_associations):

    n = shape(measurements, 2)
    M = shape(S, 2)
    N = shape(weights, 2)

    nu = zeros((2, M))
    psi = zeros((1, M))

    for j in known_associations:
        '''
        z_i = tile(measurements[:, i], (1, M, N)) 
        nu[:, :, :] = z_i - z_hat
        nu[2, :, :] = mod(nu[2, :, :] + pi, 2*pi) - pi
        q = flip(tile(diag(Q), [1, M, N]), axis=1)
        d = sum(nu**2/q)  # Assuming Q is 2x2
        psi[:, :] = 1/(2*pi*linalg.det(Q)**.5)
        Psi[i, :] = max(psi, [], [2])
        '''
        z_hat = measurement_model(S, W, j)
        nu[:, :] = measurements[:, j] - z_hat
        nu[2, :] = mod(nu[2, :] + pi, 2 * pi) - pi
        q = tile(flip(array([diag(Q)]).T, axis=1), [1, M])
        d = sum(nu ** 2 / q)  # Assuming Q is 2x2
        psi[j, :] = 1 / (2 * pi * linalg.det(Q) ** .5)*exp(-.5*d)

    reshape(psi, (1, n, M))
    outlier = mean(psi, axis=2) <= lambda_Psi

    return outlier, psi

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

def measurement_model(S, W):
    # W is the location of the landmarks on each particles map. Shape [2*landmarks, particles]
    # S is the particle set. Shape [4, particles]
    # h is predicted measurements. Shape [2*landmarks, particles]
    no_landmarks = W.shape[0]/2
    xindices = arange(0, no_landmarks,2)
    yindices = xindices + 1

    # inputa varannan
    h = array([[sqrt(square(W[xindices] - S[0,:]) + square(W[yindices] - S[1,:]) )], [arctan2(W[yindices] - S[1,:], W[xindices] - S[0,:]) - S[2,:]]]).T

    h[1,:] = mod(h[1,:] + pi, 2 * pi) - pi;

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

